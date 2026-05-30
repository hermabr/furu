#!/usr/bin/env bash
set -euo pipefail

export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

cleanup() {
    local status=$?
    set +e
    scancel --user=furu >/dev/null 2>&1
    pkill slurmd >/dev/null 2>&1
    pkill slurmctld >/dev/null 2>&1
    pkill munged >/dev/null 2>&1
    exit "$status"
}
trap cleanup EXIT INT TERM

write_munge_key() {
    if [[ ! -s /etc/munge/munge.key ]]; then
        dd if=/dev/urandom of=/etc/munge/munge.key bs=1024 count=1 status=none
    fi
    chown munge:munge /etc/munge/munge.key
    chmod 0400 /etc/munge/munge.key
}

write_slurm_config() {
    local node_name cpu_count real_memory
    node_name="$(hostname -s)"
    cpu_count="${SLURM_CPUS:-$(nproc)}"
    real_memory="${SLURM_REAL_MEMORY:-1024}"

    mkdir -p \
        /etc/slurm \
        /sys/fs/cgroup/system.slice \
        /var/log/slurm \
        /var/spool/slurmctld \
        /var/spool/slurmd
    chown -R slurm:slurm /var/log/slurm /var/spool/slurmctld /var/spool/slurmd

    cat > /etc/slurm/slurm.conf <<EOF
ClusterName=furu-docker
SlurmctldHost=${node_name}
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldLogFile=/var/log/slurm/slurmctld.log
SlurmdLogFile=/var/log/slurm/slurmd.log
AuthType=auth/munge
CryptoType=crypto/munge
MpiDefault=none
ProctrackType=proctrack/linuxproc
ReturnToService=2
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_CPU
TaskPlugin=task/none
JobAcctGatherType=jobacct_gather/none
AccountingStorageType=accounting_storage/none
InactiveLimit=0
KillWait=5
MinJobAge=300
SlurmctldTimeout=30
SlurmdTimeout=30
Waittime=0
NodeName=${node_name} NodeAddr=127.0.0.1 CPUs=${cpu_count} Boards=1 SocketsPerBoard=1 CoresPerSocket=${cpu_count} ThreadsPerCore=1 RealMemory=${real_memory} State=UNKNOWN
PartitionName=debug Nodes=${node_name} Default=YES MaxTime=INFINITE State=UP
EOF

    cat > /etc/slurm/cgroup.conf <<EOF
CgroupPlugin=cgroup/v2
IgnoreSystemd=yes
EOF
}

start_slurm() {
    mkdir -p /run/munge /var/log/munge
    chown munge:munge /run/munge /var/log/munge
    chmod 0755 /run/munge

    munged --force
    wait_for_munge
    slurmctld -Dvvv >/var/log/slurm/slurmctld.console.log 2>&1 &
    slurmd -Dvvv >/var/log/slurm/slurmd.console.log 2>&1 &
}

wait_for_munge() {
    local deadline
    deadline=$((SECONDS + 10))
    while (( SECONDS < deadline )); do
        if [[ -S /run/munge/munge.socket.2 ]] && munge --no-input >/dev/null 2>&1; then
            return 0
        fi
        sleep 0.1
    done

    echo "Munge did not become ready within 10 seconds." >&2
    cat /var/log/munge/munged.log >&2 || true
    return 1
}

wait_for_slurm() {
    local node_name deadline state
    node_name="$(hostname -s)"
    deadline=$((SECONDS + 30))
    while (( SECONDS < deadline )); do
        if sinfo --noheader --format="%T" >/tmp/furu-sinfo.out 2>/tmp/furu-sinfo.err; then
            scontrol update "NodeName=${node_name}" State=RESUME >/dev/null 2>&1 || true
            state="$(head -n 1 /tmp/furu-sinfo.out)"
            case "$state" in
                idle|allocated|mixed)
                    return 0
                    ;;
            esac
        fi
        sleep 1
    done

    echo "Slurm did not become ready within 30 seconds." >&2
    echo "--- sinfo stderr ---" >&2
    cat /tmp/furu-sinfo.err >&2 || true
    echo "--- slurmctld log ---" >&2
    cat /var/log/slurm/slurmctld.log /var/log/slurm/slurmctld.console.log >&2 || true
    echo "--- slurmd log ---" >&2
    cat /var/log/slurm/slurmd.log /var/log/slurm/slurmd.console.log >&2 || true
    return 1
}

write_munge_key
write_slurm_config
start_slurm
wait_for_slurm

chown -R furu:furu /tmp/furu-venv /tmp/uv-cache
gosu furu "$@"
