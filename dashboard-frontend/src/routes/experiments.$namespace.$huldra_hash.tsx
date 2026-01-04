import { createFileRoute, Link } from "@tanstack/react-router";
import { useGetExperimentApiExperimentsNamespaceHuldraHashGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

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
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Metadata</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          {gitCommit ? (
            <div>
              <span className="text-muted-foreground block">Git Commit</span>
              <code className="font-mono text-xs">{gitCommit}</code>
            </div>
          ) : null}
          {gitBranch ? (
            <div>
              <span className="text-muted-foreground block">Git Branch</span>
              <span>{gitBranch}</span>
            </div>
          ) : null}
          {hostname ? (
            <div>
              <span className="text-muted-foreground block">Hostname</span>
              <span>{hostname}</span>
            </div>
          ) : null}
          {user ? (
            <div>
              <span className="text-muted-foreground block">User</span>
              <span>{user}</span>
            </div>
          ) : null}
        </div>
        {pythonDef ? (
          <div className="mt-4 border-t pt-4">
            <span className="mb-2 block text-sm text-muted-foreground">
              Python Definition
            </span>
            <pre className="rounded-lg bg-muted p-4 text-sm overflow-x-auto">
              <code className="text-emerald-300">{pythonDef}</code>
            </pre>
          </div>
        ) : null}
      </CardContent>
    </Card>
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
        <div className="text-muted-foreground">Loading experiment...</div>
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="max-w-5xl mx-auto">
        <Card className="border-destructive/40">
          <CardContent className="p-6 text-center">
            <h2 className="mb-2 text-xl font-bold text-destructive">
              Experiment Not Found
            </h2>
            <p className="mb-4 text-muted-foreground">
              The experiment you're looking for doesn't exist or has been removed.
            </p>
            <Button asChild variant="ghost">
              <Link to="/experiments">← Back to experiments</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      {/* Breadcrumb */}
      <div className="mb-6 text-sm">
        <Link
          to="/experiments"
          className="text-muted-foreground hover:text-primary"
        >
          Experiments
        </Link>
        <span className="mx-2 text-muted-foreground">/</span>
        <span className="text-foreground">{experiment.class_name}</span>
      </div>

      {/* Header */}
      <Card className="mb-6">
        <CardHeader className="pb-4">
          <div className="flex items-start justify-between">
            <div>
              <CardTitle className="text-2xl">{experiment.class_name}</CardTitle>
              <p className="text-sm font-mono text-muted-foreground">
                {experiment.namespace}
              </p>
            </div>
            <div className="flex gap-2">
              <StatusBadge status={experiment.result_status} type="result" />
              {experiment.attempt_status ? (
                <StatusBadge status={experiment.attempt_status} type="attempt" />
              ) : null}
            </div>
          </div>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
            <div>
              <span className="text-muted-foreground block">Hash</span>
              <code className="font-mono">{experiment.huldra_hash}</code>
            </div>
            <div>
              <span className="text-muted-foreground block">Attempt #</span>
              <span>{experiment.attempt_number ?? "—"}</span>
            </div>
            <div>
              <span className="text-muted-foreground block">Started</span>
              <span>
                {experiment.started_at
                  ? new Date(experiment.started_at).toLocaleString()
                  : "—"}
              </span>
            </div>
            <div>
              <span className="text-muted-foreground block">Updated</span>
              <span>
                {experiment.updated_at
                  ? new Date(experiment.updated_at).toLocaleString()
                  : "—"}
              </span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Attempt Details */}
      {experiment.attempt && (
        <Card className="mb-6">
          <CardHeader>
            <CardTitle>Current Attempt</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="text-muted-foreground block">ID</span>
                <code className="font-mono text-xs">
                  {experiment.attempt.id}
                </code>
              </div>
              <div>
                <span className="text-muted-foreground block">Backend</span>
                <span>{experiment.attempt.backend}</span>
              </div>
              <div>
                <span className="text-muted-foreground block">Status</span>
                <StatusBadge status={experiment.attempt.status} type="attempt" />
              </div>
              <div>
                <span className="text-muted-foreground block">Host</span>
                <span>{experiment.attempt.owner?.host ?? "—"}</span>
              </div>
              <div>
                <span className="text-muted-foreground block">PID</span>
                <span className="font-mono">
                  {experiment.attempt.owner?.pid ?? "—"}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground block">User</span>
                <span>{experiment.attempt.owner?.user ?? "—"}</span>
              </div>
              <div>
                <span className="text-muted-foreground block">Started At</span>
                <span>
                  {new Date(experiment.attempt.started_at).toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground block">
                  Last Heartbeat
                </span>
                <span>
                  {new Date(experiment.attempt.heartbeat_at).toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-muted-foreground block">
                  Lease Expires
                </span>
                <span>
                  {new Date(experiment.attempt.lease_expires_at).toLocaleString()}
                </span>
              </div>
            </div>
            {experiment.attempt.reason ? (
              <div className="mt-4 border-t pt-4">
                <span className="block text-sm text-muted-foreground">
                  Reason
                </span>
                <span className="text-amber-300">
                  {experiment.attempt.reason}
                </span>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Metadata */}
      {experiment.metadata && (
        <MetadataSection
          metadata={experiment.metadata as Record<string, unknown>}
        />
      )}

      {/* Directory */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Directory</CardTitle>
        </CardHeader>
        <CardContent>
          <code className="break-all font-mono text-sm text-muted-foreground">
            {experiment.directory}
          </code>
        </CardContent>
      </Card>

      {/* Raw State JSON */}
      <details className="rounded-lg border bg-card">
        <summary className="cursor-pointer px-6 py-4 text-muted-foreground hover:text-foreground">
          View Raw State JSON
        </summary>
        <div className="px-6 pb-6">
          <pre className="max-h-96 overflow-x-auto rounded-lg bg-muted p-4 text-sm">
            <code className="text-muted-foreground">
              {JSON.stringify(experiment.state, null, 2)}
            </code>
          </pre>
        </div>
      </details>
    </div>
  );
}
