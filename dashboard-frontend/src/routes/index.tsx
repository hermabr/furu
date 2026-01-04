import { createFileRoute, Link } from "@tanstack/react-router";
import { useHealthCheckApiHealthGet } from "../api/endpoints/api/api";
import { useDashboardStatsApiStatsGet } from "../api/endpoints/api/api";
import { useListExperimentsApiExperimentsGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";
import { StatsCard } from "../components/StatsCard";

export const Route = createFileRoute("/")({
  component: HomePage,
});

function HomePage() {
  const { data: health, isLoading: healthLoading } =
    useHealthCheckApiHealthGet();
  const { data: stats, isLoading: statsLoading } =
    useDashboardStatsApiStatsGet();
  const { data: recentExperiments, isLoading: experimentsLoading } =
    useListExperimentsApiExperimentsGet({ limit: 5 });

  return (
    <div className="max-w-6xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Dashboard</h1>
        <p className="text-slate-400">
          Monitor your Huldra experiments in real-time
        </p>
      </div>

      {/* Health Status */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 mb-8">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-slate-400">API Status:</span>
            {healthLoading ? (
              <span className="text-slate-500">Checking...</span>
            ) : health?.status === "healthy" ? (
              <span className="text-emerald-400 font-medium flex items-center gap-2">
                <span className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></span>
                Healthy
              </span>
            ) : (
              <span className="text-red-400 font-medium">Disconnected</span>
            )}
          </div>
          <span className="text-slate-500 text-sm font-mono">
            v{health?.version || "..."}
          </span>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="Total Experiments"
          value={stats?.total ?? 0}
          loading={statsLoading}
          icon="📊"
        />
        <StatsCard
          title="Running"
          value={stats?.running_count ?? 0}
          loading={statsLoading}
          variant="running"
          icon="🔄"
        />
        <StatsCard
          title="Successful"
          value={stats?.success_count ?? 0}
          loading={statsLoading}
          variant="success"
          icon="✓"
        />
        <StatsCard
          title="Failed"
          value={stats?.failed_count ?? 0}
          loading={statsLoading}
          variant="failed"
          icon="✗"
        />
      </div>

      {/* Status Distribution */}
      {stats && stats.by_result_status && stats.by_result_status.length > 0 && (
        <div className="bg-slate-900 border border-slate-800 rounded-lg p-6 mb-8">
          <h2 className="text-lg font-semibold text-white mb-4">
            Result Status Distribution
          </h2>
          <div className="flex gap-4 flex-wrap">
            {stats.by_result_status.map((item) => (
              <div
                key={item.status}
                className="flex items-center gap-2 bg-slate-800 px-3 py-2 rounded-lg"
              >
                <StatusBadge status={item.status} type="result" />
                <span className="text-white font-mono">{item.count}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Experiments */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-slate-800 flex justify-between items-center">
          <h2 className="text-lg font-semibold text-white">
            Recent Experiments
          </h2>
          <Link
            to="/experiments"
            className="text-emerald-400 hover:text-emerald-300 text-sm"
          >
            View all →
          </Link>
        </div>
        {experimentsLoading ? (
          <div className="p-6 text-slate-500">Loading...</div>
        ) : recentExperiments?.experiments.length === 0 ? (
          <div className="p-6 text-slate-500">No experiments yet</div>
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-left text-slate-500 text-sm border-b border-slate-800">
                <th className="px-6 py-3 font-medium">Class</th>
                <th className="px-6 py-3 font-medium">Namespace</th>
                <th className="px-6 py-3 font-medium">Result</th>
                <th className="px-6 py-3 font-medium">Attempt</th>
                <th className="px-6 py-3 font-medium">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {recentExperiments?.experiments.map((exp) => (
                <tr
                  key={`${exp.namespace}-${exp.huldra_hash}`}
                  className="hover:bg-slate-800/50 transition-colors"
                >
                  <td className="px-6 py-4">
                    <Link
                      to="/experiments/$namespace/$huldra_hash"
                      params={{
                        namespace: exp.namespace,
                        huldra_hash: exp.huldra_hash,
                      }}
                      className="text-white font-medium hover:text-emerald-400"
                    >
                      {exp.class_name}
                    </Link>
                  </td>
                  <td className="px-6 py-4 text-slate-400 font-mono text-sm">
                    {exp.namespace}
                  </td>
                  <td className="px-6 py-4">
                    <StatusBadge status={exp.result_status} type="result" />
                  </td>
                  <td className="px-6 py-4">
                    {exp.attempt_status ? (
                      <StatusBadge status={exp.attempt_status} type="attempt" />
                    ) : (
                      <span className="text-slate-600">—</span>
                    )}
                  </td>
                  <td className="px-6 py-4 text-slate-500 text-sm">
                    {exp.updated_at
                      ? new Date(exp.updated_at).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}



