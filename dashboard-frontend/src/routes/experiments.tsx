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
        <h1 className="text-3xl font-bold mb-2">Experiments</h1>
        <p className="text-muted-foreground">
          Browse and filter all Huldra experiments
        </p>
      </div>

      {/* Filters */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Filters</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-4">
            <div className="flex-1 min-w-[200px]">
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
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {RESULT_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status || "All Results"}
                  </option>
                ))}
              </select>
            </div>
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
                className="h-10 rounded-md border border-input bg-background px-3 py-2 text-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
              >
                {ATTEMPT_STATUSES.map((status) => (
                  <option key={status} value={status}>
                    {status || "All Attempts"}
                  </option>
                ))}
              </select>
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
                <TableHead>Attempt</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {data?.experiments.map((exp) => (
                <TableRow key={`${exp.namespace}-${exp.huldra_hash}`}>
                  <TableCell className="pl-6">
                    <Link
                      to="/experiments/$namespace/$huldra_hash"
                      params={{
                        namespace: exp.namespace,
                        huldra_hash: exp.huldra_hash,
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
                      {exp.huldra_hash.slice(0, 8)}...
                    </code>
                  </TableCell>
                  <TableCell>
                    <StatusBadge status={exp.result_status} type="result" />
                  </TableCell>
                  <TableCell>
                    {exp.attempt_status ? (
                      <StatusBadge status={exp.attempt_status} type="attempt" />
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </TableCell>
                  <TableCell className="whitespace-nowrap text-sm text-muted-foreground">
                    {exp.updated_at
                      ? new Date(exp.updated_at).toLocaleString()
                      : "—"}
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
