import { createFileRoute, Link } from "@tanstack/react-router";
import { useState } from "react";
import { useListExperimentsApiExperimentsGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";
import { EmptyState } from "../components/EmptyState";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { Input } from "../components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";

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
const BACKENDS = ["", "local", "submitit"] as const;
const SCHEMA_FILTERS = ["current", "stale", "any"] as const;

/**
 * Render the Experiments dashboard with filter controls, results listing, and pagination.
 *
 * The page exposes filters for namespace, result/attempt status, backend, hostname, user, schema,
 * start time range, and config; it displays matching experiments in a table with status badges,
 * migration and attempt indicators, and navigation controls when multiple pages exist.
 *
 * @returns The React element for the Experiments page including filter UI, results table, and pagination.
 */
function ExperimentsPage() {
  const [resultFilter, setResultFilter] = useState("");
  const [attemptFilter, setAttemptFilter] = useState("");
  const [namespaceFilter, setNamespaceFilter] = useState("");
  const [backendFilter, setBackendFilter] = useState("");
  const [hostnameFilter, setHostnameFilter] = useState("");
  const [userFilter, setUserFilter] = useState("");
  const [schemaFilter, setSchemaFilter] = useState<
    (typeof SCHEMA_FILTERS)[number]
  >("current");
  const [startedAfter, setStartedAfter] = useState("");
  const [startedBefore, setStartedBefore] = useState("");
  const [configFilter, setConfigFilter] = useState("");
  const [viewMode, setViewMode] = useState("resolved");
  const [page, setPage] = useState(0);
  const limit = 20;

  const { data, isLoading, error } = useListExperimentsApiExperimentsGet({
    result_status: resultFilter || undefined,
    attempt_status: attemptFilter || undefined,
    namespace: namespaceFilter || undefined,
    backend: backendFilter || undefined,
    hostname: hostnameFilter || undefined,
    user: userFilter || undefined,
    schema: schemaFilter,
    started_after: startedAfter || undefined,
    started_before: startedBefore || undefined,
    config_filter: configFilter || undefined,
    view: viewMode,
    limit,
    offset: page * limit,
  });

  const totalPages = data ? Math.ceil(data.total / limit) : 0;

  const resetFilters = () => {
    setResultFilter("");
    setAttemptFilter("");
    setNamespaceFilter("");
    setBackendFilter("");
    setHostnameFilter("");
    setUserFilter("");
    setSchemaFilter("current");
    setStartedAfter("");
    setStartedBefore("");
    setConfigFilter("");
    setPage(0);
  };

  const hasFilters =
    resultFilter ||
    attemptFilter ||
    namespaceFilter ||
    backendFilter ||
    hostnameFilter ||
    userFilter ||
    schemaFilter !== "current" ||
    startedAfter ||
    startedBefore ||
    configFilter;

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Experiments</h1>
        <p className="text-muted-foreground">
          Browse and filter all Furu experiments
        </p>
      </div>

      {/* Filters */}
      <Card className="mb-6">
          <CardHeader className="flex flex-row items-center justify-between">
            <CardTitle>Filters</CardTitle>
            <div className="flex items-center gap-2">
              <div className="flex rounded-full border border-muted overflow-hidden text-xs">
                <button
                  className={`px-3 py-1 ${
                    viewMode === "resolved"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground"
                  }`}
                  onClick={() => {
                    setViewMode("resolved");
                    setPage(0);
                  }}
                >
                  Aliased
                </button>
                <button
                  className={`px-3 py-1 ${
                    viewMode === "original"
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground"
                  }`}
                  onClick={() => {
                    setViewMode("original");
                    setPage(0);
                  }}
                >
                  Original
                </button>
              </div>
              {hasFilters && (
                <Button variant="ghost" size="sm" onClick={resetFilters}>
                  Clear filters
                </Button>
              )}
            </div>
          </CardHeader>

        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
            {/* Namespace filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Namespace
              </label>
              <Input
                type="text"
                placeholder="Filter by namespace..."
                value={namespaceFilter}
                onChange={(e) => {
                  setNamespaceFilter(e.target.value);
                  setPage(0);
                }}
              />
            </div>

            {/* Result status filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Result Status
              </label>
              <select
                value={resultFilter}
                onChange={(e) => {
                  setResultFilter(e.target.value);
                  setPage(0);
                }}
                className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {RESULT_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status || "All Results"}
                  </option>
                ))}
              </select>
            </div>

            {/* Attempt status filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Attempt Status
              </label>
              <select
                value={attemptFilter}
                onChange={(e) => {
                  setAttemptFilter(e.target.value);
                  setPage(0);
                }}
                className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {ATTEMPT_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status || "All Attempts"}
                  </option>
                ))}
              </select>
            </div>

            {/* Backend filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Backend
              </label>
              <select
                value={backendFilter}
                onChange={(e) => {
                  setBackendFilter(e.target.value);
                  setPage(0);
                }}
                className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {BACKENDS.map((backend) => (
                  <option key={backend} value={backend}>
                    {backend || "All Backends"}
                  </option>
                ))}
              </select>
            </div>

            {/* Hostname filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Hostname
              </label>
              <Input
                type="text"
                placeholder="Filter by hostname..."
                value={hostnameFilter}
                onChange={(e) => {
                  setHostnameFilter(e.target.value);
                  setPage(0);
                }}
              />
            </div>

            {/* User filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                User
              </label>
              <Input
                type="text"
                placeholder="Filter by user..."
                value={userFilter}
                onChange={(e) => {
                  setUserFilter(e.target.value);
                  setPage(0);
                }}
              />
            </div>

            {/* Schema filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Schema
              </label>
              <select
                value={schemaFilter}
                onChange={(e) => {
                  setSchemaFilter(e.target.value as (typeof SCHEMA_FILTERS)[number]);
                  setPage(0);
                }}
                className="h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {SCHEMA_FILTERS.map((schema) => (
                  <option key={schema} value={schema}>
                    {schema === "current"
                      ? "Current"
                      : schema === "stale"
                        ? "Stale"
                        : "Any"}
                  </option>
                ))}
              </select>
            </div>

            {/* Started after filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Started After
              </label>
              <Input
                type="datetime-local"
                value={startedAfter}
                onChange={(e) => {
                  setStartedAfter(e.target.value ? e.target.value + ":00" : "");
                  setPage(0);
                }}
              />
            </div>

            {/* Started before filter */}
            <div>
              <label className="mb-1 block text-sm text-muted-foreground">
                Started Before
              </label>
              <Input
                type="datetime-local"
                value={startedBefore}
                onChange={(e) => {
                  setStartedBefore(e.target.value ? e.target.value + ":00" : "");
                  setPage(0);
                }}
              />
            </div>

            {/* Config filter */}
            <div className="md:col-span-2">
              <label className="mb-1 block text-sm text-muted-foreground">
                Config Filter
              </label>
              <Input
                type="text"
                placeholder="field.path=value (e.g., model.name=gpt-4)"
                value={configFilter}
                onChange={(e) => {
                  setConfigFilter(e.target.value);
                  setPage(0);
                }}
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Results count */}
      <div className="mb-4 text-sm text-muted-foreground">
        {data ? (
          <>
            Showing {data.experiments.length} of {data.total} experiments
          </>
        ) : (
          "Loading..."
        )}
      </div>

      {/* Table */}
      {isLoading ? (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            Loading experiments...
          </CardContent>
        </Card>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center text-destructive">
            Error loading experiments. Is the API running?
          </CardContent>
        </Card>
      ) : data?.experiments.length === 0 ? (
        <EmptyState
          title="No experiments found"
          description="Try adjusting your filters or create some experiments first."
          icon="🔬"
        />
      ) : (
        <Card className="overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Class</TableHead>
                <TableHead>Namespace</TableHead>
                <TableHead>Hash</TableHead>
                <TableHead>Result</TableHead>
                <TableHead>Migration</TableHead>
                <TableHead>Attempt</TableHead>

                <TableHead>Backend</TableHead>
                <TableHead>Host</TableHead>
                <TableHead>User</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.experiments.map((exp) => (
                <TableRow key={`${exp.namespace}-${exp.furu_hash}`}>
                  <TableCell className="pl-6">
                    <Link
                      to="/experiments/$namespace/$furu_hash"
                      params={{
                        namespace: exp.namespace,
                        furu_hash: exp.furu_hash,
                      }}
                      className="font-medium hover:text-primary"
                    >
                      {exp.class_name}
                    </Link>
                  </TableCell>
                  <TableCell className="max-w-xs truncate font-mono text-sm text-muted-foreground">
                    {exp.namespace}
                  </TableCell>
                  <TableCell>
                    <code className="rounded bg-muted px-2 py-1 font-mono text-xs text-muted-foreground">
                      {exp.furu_hash.slice(0, 8)}...
                    </code>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={exp.result_status} type="result" />
                  </TableCell>
                  <TableCell>
                    {exp.migration_kind ? (
                      <span className="text-xs rounded-full border border-amber-500/40 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                        {exp.migration_kind}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                  <TableCell>
                    {exp.attempt_status ? (
                      <StatusBadge status={exp.attempt_status} type="attempt" />
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {exp.backend || "-"}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground max-w-[150px] truncate" title={exp.hostname || undefined}>
                    {exp.hostname || "-"}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {exp.user || "-"}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-sm text-muted-foreground">
                    {exp.updated_at
                      ? new Date(exp.updated_at).toLocaleString()
                      : "-"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>

          {totalPages > 1 && (
            <CardContent className="flex items-center justify-between border-t py-4">
              <Button
                variant="secondary"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={page === 0}
              >
                Previous
              </Button>
              <span className="text-sm text-muted-foreground">
                Page {page + 1} of {totalPages}
              </span>
              <Button
                variant="secondary"
                onClick={() =>
                  setPage((p) => Math.min(totalPages - 1, p + 1))
                }
                disabled={page >= totalPages - 1}
              >
                Next
              </Button>
            </CardContent>
          )}
        </Card>
      )}
    </div>
  );
}