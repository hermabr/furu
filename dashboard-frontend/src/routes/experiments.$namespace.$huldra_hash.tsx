import { createFileRoute, Link } from "@tanstack/react-router";
import { useGetExperimentApiExperimentsNamespaceHuldraHashGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";

function MetadataSection({ metadata }: { metadata: Record<string, unknown> }) {
  const getString = (key: string): string | null => {
    const value = metadata[key];
    return typeof value === "string" ? value : null;
  };

  const gitCommit = getString("git_commit");
  const gitBranch = getString("git_branch");
  const hostname = getString("hostname");
  const user = getString("user");
  const pythonDef = getString("huldra_python_def");

  return (
    <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-6">
      <h2 className="text-lg font-semibold text-white mb-4">Metadata</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
        {gitCommit && (
          <div>
            <span className="text-slate-500 block">Git Commit</span>
            <code className="text-white font-mono text-xs">{gitCommit}</code>
          </div>
        )}
        {gitBranch && (
          <div>
            <span className="text-slate-500 block">Git Branch</span>
            <span className="text-white">{gitBranch}</span>
          </div>
        )}
        {hostname && (
          <div>
            <span className="text-slate-500 block">Hostname</span>
            <span className="text-white">{hostname}</span>
          </div>
        )}
        {user && (
          <div>
            <span className="text-slate-500 block">User</span>
            <span className="text-white">{user}</span>
          </div>
        )}
      </div>
      {pythonDef && (
        <div className="mt-4 pt-4 border-t border-slate-800">
          <span className="text-slate-500 block text-sm mb-2">
            Python Definition
          </span>
          <pre className="bg-slate-950 rounded-lg p-4 overflow-x-auto text-sm">
            <code className="text-emerald-300">{pythonDef}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

export const Route = createFileRoute("/experiments/$namespace/$huldra_hash")({
  component: ExperimentDetailPage,
});

function ExperimentDetailPage() {
  const { namespace, huldra_hash } = Route.useParams();
  const {
    data: experiment,
    isLoading,
    error,
  } = useGetExperimentApiExperimentsNamespaceHuldraHashGet(namespace, huldra_hash);

  if (isLoading) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="text-slate-500">Loading experiment...</div>
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="max-w-5xl mx-auto">
        <div className="bg-red-900/20 border border-red-800 rounded-lg p-6 text-center">
          <h2 className="text-xl font-bold text-red-400 mb-2">
            Experiment Not Found
          </h2>
          <p className="text-slate-400 mb-4">
            The experiment you're looking for doesn't exist or has been removed.
          </p>
          <Link
            to="/experiments"
            className="text-emerald-400 hover:text-emerald-300"
          >
            ← Back to experiments
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Breadcrumb */}
      <div className="mb-6 text-sm">
        <Link
          to="/experiments"
          className="text-slate-400 hover:text-emerald-400"
        >
          Experiments
        </Link>
        <span className="text-slate-600 mx-2">/</span>
        <span className="text-slate-300">{experiment.class_name}</span>
      </div>

      {/* Header */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-2xl font-bold text-white mb-1">
              {experiment.class_name}
            </h1>
            <p className="text-slate-400 font-mono text-sm">
              {experiment.namespace}
            </p>
          </div>
          <div className="flex gap-2">
            <StatusBadge status={experiment.result_status} type="result" />
            {experiment.attempt_status && (
              <StatusBadge status={experiment.attempt_status} type="attempt" />
            )}
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
          <div>
            <span className="text-slate-500 block">Hash</span>
            <code className="text-white font-mono">{experiment.huldra_hash}</code>
          </div>
          <div>
            <span className="text-slate-500 block">Attempt #</span>
            <span className="text-white">
              {experiment.attempt_number ?? "—"}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block">Started</span>
            <span className="text-white">
              {experiment.started_at
                ? new Date(experiment.started_at).toLocaleString()
                : "—"}
            </span>
          </div>
          <div>
            <span className="text-slate-500 block">Updated</span>
            <span className="text-white">
              {experiment.updated_at
                ? new Date(experiment.updated_at).toLocaleString()
                : "—"}
            </span>
          </div>
        </div>
      </div>

      {/* Attempt Details */}
      {experiment.attempt && (
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4">
            Current Attempt
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-slate-500 block">ID</span>
              <code className="text-white font-mono text-xs">
                {experiment.attempt.id}
              </code>
            </div>
            <div>
              <span className="text-slate-500 block">Backend</span>
              <span className="text-white">{experiment.attempt.backend}</span>
            </div>
            <div>
              <span className="text-slate-500 block">Status</span>
              <StatusBadge status={experiment.attempt.status} type="attempt" />
            </div>
            <div>
              <span className="text-slate-500 block">Host</span>
              <span className="text-white">
                {experiment.attempt.owner?.host ?? "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">PID</span>
              <span className="text-white font-mono">
                {experiment.attempt.owner?.pid ?? "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">User</span>
              <span className="text-white">
                {experiment.attempt.owner?.user ?? "—"}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">Started At</span>
              <span className="text-white">
                {new Date(experiment.attempt.started_at).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">Last Heartbeat</span>
              <span className="text-white">
                {new Date(experiment.attempt.heartbeat_at).toLocaleString()}
              </span>
            </div>
            <div>
              <span className="text-slate-500 block">Lease Expires</span>
              <span className="text-white">
                {new Date(experiment.attempt.lease_expires_at).toLocaleString()}
              </span>
            </div>
          </div>
          {experiment.attempt.reason && (
            <div className="mt-4 pt-4 border-t border-slate-800">
              <span className="text-slate-500 block text-sm">Reason</span>
              <span className="text-amber-400">
                {experiment.attempt.reason}
              </span>
            </div>
          )}
        </div>
      )}

      {/* Metadata */}
      {experiment.metadata && (
        <MetadataSection
          metadata={experiment.metadata as Record<string, unknown>}
        />
      )}

      {/* Directory */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-6">
        <h2 className="text-lg font-semibold text-white mb-4">Directory</h2>
        <code className="text-slate-300 font-mono text-sm break-all">
          {experiment.directory}
        </code>
      </div>

      {/* Raw State JSON */}
      <details className="bg-slate-900 border border-slate-800 rounded-lg">
        <summary className="px-6 py-4 cursor-pointer text-slate-400 hover:text-white">
          View Raw State JSON
        </summary>
        <div className="px-6 pb-6">
          <pre className="bg-slate-950 rounded-lg p-4 overflow-x-auto text-sm max-h-96">
            <code className="text-slate-300">
              {JSON.stringify(experiment.state, null, 2)}
            </code>
          </pre>
        </div>
      </details>
    </div>
  );
}
