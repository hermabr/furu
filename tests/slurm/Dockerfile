FROM python:3.12-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        gosu \
        munge \
        procps \
        slurm-wlm \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir uv

RUN getent group slurm >/dev/null || groupadd --system slurm \
    && getent passwd slurm >/dev/null || useradd --system --gid slurm --home-dir /var/lib/slurm --shell /usr/sbin/nologin slurm \
    && useradd --create-home --shell /bin/bash furu \
    && mkdir -p \
        /etc/slurm \
        /home/furu \
        /run/munge \
        /tmp/furu-venv \
        /tmp/uv-cache \
        /var/log/munge \
        /var/log/slurm \
        /var/spool/slurmctld \
        /var/spool/slurmd \
    && chown -R furu:furu /home/furu /tmp/furu-venv /tmp/uv-cache \
    && chown -R munge:munge /run/munge /etc/munge /var/log/munge \
    && chown -R slurm:slurm /var/log/slurm /var/spool/slurmctld /var/spool/slurmd

COPY tests/slurm/entrypoint.sh /usr/local/bin/furu-slurm-entrypoint
RUN chmod 0755 /usr/local/bin/furu-slurm-entrypoint

WORKDIR /workspace
ENTRYPOINT ["furu-slurm-entrypoint"]
CMD ["uv", "run", "pytest", "-p", "no:cacheprovider", "-m", "slurm_integration", "tests/test_slurm_integration.py", "-vv"]
