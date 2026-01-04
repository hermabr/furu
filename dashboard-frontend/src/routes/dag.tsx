import { createFileRoute } from "@tanstack/react-router";
import { useExperimentDagApiDagGet } from "../api/endpoints/api/api";
import { DAGVisualization } from "../components/DAGVisualization";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import { StatsCard } from "../components/StatsCard";
import { EmptyState } from "../components/EmptyState";

export const Route = createFileRoute("/dag")({
  component: DAGPage,
});

function DAGPage() {
  const { data, isLoading, error } = useExperimentDagApiDagGet();

  return (
    <div className="max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Experiment DAG</h1>
        <p className="text-muted-foreground">
          Visualize the dependency graph of your experiments. Nodes represent
          experiment classes, edges show dependencies.
        </p>
      </div>

      {/* Stats */}
      {data && (
        <div className="grid grid-cols-3 gap-4 mb-6">
          <StatsCard
            title="Classes"
            value={data.total_nodes}
            loading={isLoading}
            icon="N"
          />
          <StatsCard
            title="Dependencies"
            value={data.total_edges}
            loading={isLoading}
            icon="E"
          />
          <StatsCard
            title="Experiments"
            value={data.total_experiments}
            loading={isLoading}
            icon="#"
          />
        </div>
      )}

      {/* DAG Visualization */}
      {isLoading ? (
        <Card>
          <CardContent className="p-8 text-center text-muted-foreground">
            Loading experiment graph...
          </CardContent>
        </Card>
      ) : error ? (
        <Card>
          <CardContent className="p-8 text-center text-destructive">
            Error loading DAG. Is the API running?
          </CardContent>
        </Card>
      ) : data?.nodes.length === 0 ? (
        <EmptyState
          title="No experiments found"
          description="Create some experiments to see the dependency graph."
          icon="G"
        />
      ) : data ? (
        <Card>
          <CardHeader>
            <CardTitle>Dependency Graph</CardTitle>
            <p className="text-sm text-muted-foreground">
              Click on a node to see details and highlight connected edges.
              Scroll to zoom, drag to pan.
            </p>
          </CardHeader>
          <CardContent className="p-0">
            <DAGVisualization nodes={data.nodes} edges={data.edges} />
          </CardContent>
        </Card>
      ) : null}

      {/* Legend */}
      {data && data.nodes.length > 0 && (
        <Card className="mt-6">
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Legend</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-6 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-green-500 rounded" />
                <span>All successful</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-blue-500 rounded" />
                <span>Running experiments</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-red-500 rounded" />
                <span>Has failures</span>
              </div>
              <div className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-border rounded" />
                <span>Default</span>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
