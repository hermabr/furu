import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useListExperimentsApiExperimentsGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState } from "../components/EmptyState";

export const Route = createFileRoute("/experiments")({
  component: ExperimentsPage,
});

const RESULT_STATUSES = [
  "",
  "success",
  "failed",
  "incomplete",
  "absent",
] as const;
const ATTEMPT_STATUSES = [
  "",
  "running",
  "queued",
  "success",
  "failed",
  "crashed",
  "cancelled",
  "preempted",
] as const;

function ExperimentsPage() {
  const [resultFilter, setResultFilter] = useState("");
  const [attemptFilter, setAttemptFilter] = useState("");
  const [namespaceFilter, setNamespaceFilter] = useState("");
  const [page, setPage] = useState(0);
  const limit = 20;

  const { data, isLoading, error } = useListExperimentsApiExperimentsGet({
    result_status: resultFilter || undefined,
    attempt_status: attemptFilter || undefined,
    namespace: namespaceFilter || undefined,
    limit,
    offset: page * limit,
  });

  const totalPages = data ? Math.ceil(data.total / limit) : 0;

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white mb-2">Experiments</h1>
        <p className="text-slate-400">
          Browse and filter all Huldra experiments
        </p>
      </div>

      {/* Filters */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg p-4 mb-6">
        <div className="flex flex-wrap gap-4">
          <div className="flex-1 min-w-[200px]">
            <label className="block text-sm text-slate-400 mb-1">
              Namespace
            </label>
            <input
              type="text"
              placeholder="Filter by namespace..."
              value={namespaceFilter}
              onChange={(e) => {
                setNamespaceFilter(e.target.value);
                setPage(0);
              }}
              className="w-full bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white placeholder-slate-500 focus:outline-none focus:border-emerald-500"
            />
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">
              Result Status
            </label>
            <select
              value={resultFilter}
              onChange={(e) => {
                setResultFilter(e.target.value);
                setPage(0);
              }}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
            >
              {RESULT_STATUSES.map((status) => (
                <option key={status} value={status}>
                  {status || "All Results"}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-sm text-slate-400 mb-1">
              Attempt Status
            </label>
            <select
              value={attemptFilter}
              onChange={(e) => {
                setAttemptFilter(e.target.value);
                setPage(0);
              }}
              className="bg-slate-800 border border-slate-700 rounded-lg px-3 py-2 text-white focus:outline-none focus:border-emerald-500"
            >
              {ATTEMPT_STATUSES.map((status) => (
                <option key={status} value={status}>
                  {status || "All Attempts"}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Results count */}
      <div className="mb-4 text-slate-400 text-sm">
        {data ? (
          <>
            Showing {data.experiments.length} of {data.total} experiments
          </>
        ) : (
          "Loading..."
        )}
      </div>

      {/* Table */}
      <div className="bg-slate-900 border border-slate-800 rounded-lg overflow-hidden">
        {isLoading ? (
          <div className="p-8 text-center text-slate-500">
            Loading experiments...
          </div>
        ) : error ? (
          <div className="p-8 text-center text-red-400">
            Error loading experiments. Is the API running?
          </div>
        ) : data?.experiments.length === 0 ? (
          <EmptyState
            title="No experiments found"
            description="Try adjusting your filters or create some experiments first."
            icon="🔬"
          />
        ) : (
          <table className="w-full">
            <thead>
              <tr className="text-left text-slate-500 text-sm border-b border-slate-800">
                <th className="px-6 py-3 font-medium">Class</th>
                <th className="px-6 py-3 font-medium">Namespace</th>
                <th className="px-6 py-3 font-medium">Hash</th>
                <th className="px-6 py-3 font-medium">Result</th>
                <th className="px-6 py-3 font-medium">Attempt</th>
                <th className="px-6 py-3 font-medium">Updated</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {data?.experiments.map((exp) => (
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
                  <td className="px-6 py-4 text-slate-400 font-mono text-sm max-w-xs truncate">
                    {exp.namespace}
                  </td>
                  <td className="px-6 py-4">
                    <code className="text-slate-500 font-mono text-xs bg-slate-800 px-2 py-1 rounded">
                      {exp.huldra_hash.slice(0, 8)}...
                    </code>
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
                  <td className="px-6 py-4 text-slate-500 text-sm whitespace-nowrap">
                    {exp.updated_at
                      ? new Date(exp.updated_at).toLocaleString()
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="px-6 py-4 border-t border-slate-800 flex items-center justify-between">
            <button
              onClick={() => setPage((p) => Math.max(0, p - 1))}
              disabled={page === 0}
              className="px-4 py-2 bg-slate-800 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-700"
            >
              Previous
            </button>
            <span className="text-slate-400">
              Page {page + 1} of {totalPages}
            </span>
            <button
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
              disabled={page >= totalPages - 1}
              className="px-4 py-2 bg-slate-800 text-white rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-slate-700"
            >
              Next
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

