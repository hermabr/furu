import { useState } from "react";
import { createFileRoute, Link } from "@tanstack/react-router";
import {
  useGetExperimentApiExperimentsNamespaceHuldraHashGet,
  useGetExperimentRelationshipsRouteApiExperimentsNamespaceHuldraHashRelationshipsGet,
} from "../api/endpoints/api/api";
import type {
  ChildExperiment,
  ParentExperiment,
} from "../api/models";
import { StatusBadge } from "../components/StatusBadge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";

// Type for serialized huldra_obj values
type ConfigValue =
  | string
  | number
  | boolean
  | null
  | ConfigValue[]
  | { [key: string]: ConfigValue };

interface HuldraObj {
  __class__?: string;
  [key: string]: ConfigValue | undefined;
}

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

/**
 * Renders a config value, recursively handling nested objects
 */
function ConfigValueDisplay({
  value,
  depth = 0,
}: {
  value: ConfigValue;
  depth?: number;
}) {
  if (value === null) {
    return <span className="text-muted-foreground italic">null</span>;
  }

  if (typeof value === "boolean") {
    return (
      <span className={value ? "text-green-400" : "text-red-400"}>
        {String(value)}
      </span>
    );
  }

  if (typeof value === "number") {
    return <span className="text-blue-400">{value}</span>;
  }

  if (typeof value === "string") {
    return <span className="text-amber-300">"{value}"</span>;
  }

  if (Array.isArray(value)) {
    if (value.length === 0) {
      return <span className="text-muted-foreground">[]</span>;
    }
    return (
      <div className="ml-4">
        <span className="text-muted-foreground">[</span>
        {value.map((item, i) => (
          <div key={i} className="ml-4">
            <ConfigValueDisplay value={item} depth={depth + 1} />
            {i < value.length - 1 && (
              <span className="text-muted-foreground">,</span>
            )}
          </div>
        ))}
        <span className="text-muted-foreground">]</span>
      </div>
    );
  }

  if (typeof value === "object") {
    const entries = Object.entries(value);
    if (entries.length === 0) {
      return <span className="text-muted-foreground">{"{}"}</span>;
    }
    return (
      <div className={depth > 0 ? "ml-4" : ""}>
        {entries.map(([k, v], i) => (
          <div key={k} className="flex">
            <span className="text-purple-400 mr-1">{k}:</span>
            <ConfigValueDisplay value={v} depth={depth + 1} />
            {i < entries.length - 1 && (
              <span className="text-muted-foreground">,</span>
            )}
          </div>
        ))}
      </div>
    );
  }

  return <span>{String(value)}</span>;
}

/**
 * Collapsible section for a group of config fields
 */
function CollapsibleConfigSection({
  title,
  children,
  defaultOpen = false,
}: {
  title: string;
  children: React.ReactNode;
  defaultOpen?: boolean;
}) {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="border rounded-lg mb-2">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2 text-left flex items-center justify-between hover:bg-muted/50 rounded-lg"
      >
        <span className="font-medium text-sm">{title}</span>
        <span className="text-muted-foreground">{isOpen ? "−" : "+"}</span>
      </button>
      {isOpen && <div className="px-4 pb-4 pt-2">{children}</div>}
    </div>
  );
}

/**
 * Displays the experiment configuration with collapsible sections
 */
function ConfigSection({
  huldraObj,
  parents,
}: {
  huldraObj: HuldraObj;
  parents?: ParentExperiment[];
}) {
  // Separate regular config fields from nested Huldra objects (dependencies)
  const configFields: [string, ConfigValue][] = [];
  const dependencyFields: [string, HuldraObj][] = [];

  for (const [key, value] of Object.entries(huldraObj)) {
    if (key === "__class__") continue;

    if (
      value &&
      typeof value === "object" &&
      !Array.isArray(value) &&
      "__class__" in value
    ) {
      dependencyFields.push([key, value as HuldraObj]);
    } else {
      configFields.push([key, value as ConfigValue]);
    }
  }

  // Create a map of field_name -> parent for quick lookup
  const parentByField = new Map(
    parents?.map((p) => [p.field_name, p]) ?? []
  );

  const hasConfig = configFields.length > 0;
  const hasDependencies = dependencyFields.length > 0;

  if (!hasConfig && !hasDependencies) {
    return (
      <Card className="mb-6">
        <CardHeader>
          <CardTitle>Configuration</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground text-sm">
            No configuration parameters
          </p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Configuration</CardTitle>
      </CardHeader>
      <CardContent>
        {hasConfig && (
          <CollapsibleConfigSection title="Parameters" defaultOpen={true}>
            <div className="font-mono text-sm space-y-1">
              {configFields.map(([key, value]) => (
                <div key={key} className="flex">
                  <span className="text-purple-400 mr-2 min-w-fit">{key}:</span>
                  <ConfigValueDisplay value={value} />
                </div>
              ))}
            </div>
          </CollapsibleConfigSection>
        )}

        {hasDependencies && (
          <CollapsibleConfigSection title="Dependencies (Embedded Config)">
            <div className="space-y-3">
              {dependencyFields.map(([fieldName, depObj]) => {
                const parent = parentByField.get(fieldName);
                const isClickable = parent?.namespace && parent?.huldra_hash;

                return (
                  <div
                    key={fieldName}
                    className="border-l-2 border-muted pl-3"
                  >
                    <div className="text-sm mb-1 flex items-center justify-between">
                      <div className="flex items-center">
                        <span className="text-purple-400 font-medium">
                          {fieldName}
                        </span>
                        <span className="text-muted-foreground mx-1">:</span>
                        {isClickable ? (
                          <Link
                            to="/experiments/$namespace/$huldra_hash"
                            params={{
                              namespace: parent.namespace!,
                              huldra_hash: parent.huldra_hash!,
                            }}
                            className="text-cyan-400 hover:underline"
                          >
                            {depObj.__class__}
                          </Link>
                        ) : (
                          <span className="text-cyan-400">{depObj.__class__}</span>
                        )}
                      </div>
                      {parent?.result_status && (
                        <StatusBadge status={parent.result_status} type="result" />
                      )}
                    </div>
                    <div className="font-mono text-xs space-y-0.5 text-muted-foreground">
                      {Object.entries(depObj)
                        .filter(([k]) => k !== "__class__")
                        .map(([k, v]) => (
                          <div key={k} className="flex">
                            <span className="mr-1">{k}:</span>
                            <ConfigValueDisplay
                              value={v as ConfigValue}
                              depth={1}
                            />
                          </div>
                        ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </CollapsibleConfigSection>
        )}
      </CardContent>
    </Card>
  );
}

/**
 * Displays parent experiments with navigation links
 */
function ParentsSection({ parents }: { parents: ParentExperiment[] }) {
  if (parents.length === 0) return null;

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Parent Experiments</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-3">
          {parents.map((parent, index) => (
            <div
              key={`${parent.field_name}-${index}`}
              className="border rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <span className="text-purple-400 font-medium text-sm">
                    {parent.field_name}
                  </span>
                  <span className="text-muted-foreground">:</span>
                  <span className="text-cyan-400 text-sm">
                    {parent.class_name}
                  </span>
                </div>
                {parent.result_status && (
                  <StatusBadge status={parent.result_status} type="result" />
                )}
              </div>

              {parent.namespace && parent.huldra_hash ? (
                <Link
                  to="/experiments/$namespace/$huldra_hash"
                  params={{
                    namespace: parent.namespace,
                    huldra_hash: parent.huldra_hash,
                  }}
                  className="text-sm text-primary hover:underline flex items-center gap-1"
                >
                  <span>View experiment</span>
                  <span className="text-muted-foreground font-mono text-xs">
                    ({parent.huldra_hash.slice(0, 8)}...)
                  </span>
                </Link>
              ) : (
                <span className="text-sm text-muted-foreground italic">
                  Experiment not found in storage
                </span>
              )}

              {parent.config && Object.keys(parent.config).length > 0 && (
                <details className="mt-2">
                  <summary className="text-xs text-muted-foreground cursor-pointer hover:text-foreground">
                    Show config
                  </summary>
                  <div className="mt-1 font-mono text-xs pl-2 border-l border-muted">
                    {Object.entries(parent.config).map(([k, v]) => (
                      <div key={k} className="flex">
                        <span className="text-muted-foreground mr-1">{k}:</span>
                        <ConfigValueDisplay
                          value={v as ConfigValue}
                          depth={1}
                        />
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

/**
 * Displays child experiments that depend on this experiment
 */
function ChildrenSection({ children }: { children: ChildExperiment[] }) {
  if (children.length === 0) return null;

  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle>Child Experiments ({children.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="space-y-2">
          {children.map((child, index) => (
            <div
              key={`${child.namespace}-${child.huldra_hash}-${index}`}
              className="border rounded-lg p-3 flex items-center justify-between"
            >
              <div className="flex items-center gap-3">
                <div>
                  <Link
                    to="/experiments/$namespace/$huldra_hash"
                    params={{
                      namespace: child.namespace,
                      huldra_hash: child.huldra_hash,
                    }}
                    className="text-primary hover:underline font-medium"
                  >
                    {child.class_name}
                  </Link>
                  <div className="text-xs text-muted-foreground">
                    via{" "}
                    <span className="text-purple-400">{child.field_name}</span>
                  </div>
                </div>
              </div>
              <div className="flex items-center gap-2">
                <code className="text-xs text-muted-foreground font-mono">
                  {child.huldra_hash.slice(0, 8)}...
                </code>
                <StatusBadge status={child.result_status} type="result" />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

export const Route = createFileRoute("/experiments_/$namespace/$huldra_hash")({
  component: ExperimentDetailPage,
});

function ExperimentDetailPage() {
  const { namespace, huldra_hash } = Route.useParams();
  const {
    data: experiment,
    isLoading,
    error,
  } = useGetExperimentApiExperimentsNamespaceHuldraHashGet(
    namespace,
    huldra_hash
  );

  const { data: relationships } =
    useGetExperimentRelationshipsRouteApiExperimentsNamespaceHuldraHashRelationshipsGet(
      namespace,
      huldra_hash
    );

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
              The experiment you're looking for doesn't exist or has been
              removed.
            </p>
            <Button asChild variant="ghost">
              <Link to="/experiments">Back to experiments</Link>
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  // Extract huldra_obj from metadata for config display
  const huldraObj =
    experiment.metadata &&
    typeof experiment.metadata === "object" &&
    "huldra_obj" in experiment.metadata
      ? (experiment.metadata.huldra_obj as HuldraObj)
      : null;

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

      {/* Configuration Section */}
      {huldraObj && (
        <ConfigSection huldraObj={huldraObj} parents={relationships?.parents} />
      )}

      {/* Parent Experiments */}
      {relationships?.parents && relationships.parents.length > 0 && (
        <ParentsSection parents={relationships.parents} />
      )}

      {/* Child Experiments */}
      {relationships?.children && relationships.children.length > 0 && (
        <ChildrenSection children={relationships.children} />
      )}

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
