import { createFileRoute, Link } from "@tanstack/react-router";
import { useHealthCheckApiHealthGet } from "../api/endpoints/api/api";
import { useDashboardStatsApiStatsGet } from "../api/endpoints/api/api";
import { useListExperimentsApiExperimentsGet } from "../api/endpoints/api/api";
import { StatusBadge } from "../components/StatusBadge";
import { StatsCard } from "../components/StatsCard";
import { Badge } from "../components/ui/badge";
import { Button } from "../components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "../components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "../components/ui/table";

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
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-muted-foreground">
          Monitor your Huldra experiments in real-time
        </p>
      </div>

      {/* Health Status */}
      <Card className="mb-8">
        <CardContent className="flex items-center justify-between p-4">
          <div className="flex items-center gap-4">
            <span className="text-muted-foreground">API Status:</span>
            {healthLoading ? (
              <Badge variant="muted">Checking...</Badge>
            ) : health?.status === "healthy" ? (
              <Badge variant="success" className="gap-2">
                <span className="h-2 w-2 rounded-full bg-emerald-300 animate-pulse" />
                Healthy
              </Badge>
            ) : (
              <Badge variant="destructive">Disconnected</Badge>
            )}
          </div>
          <span className="text-muted-foreground text-sm font-mono">
            v{health?.version || "..."}
          </span>
        </CardContent>
      </Card>

      {/* Stats Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
        <StatsCard
          title="Total Experiments"
          value={stats?.total ?? 0}
          loading={statsLoading}
          icon="📊"
          testId="stats-total"
        />
        <StatsCard
          title="Running"
          value={stats?.running_count ?? 0}
          loading={statsLoading}
          variant="running"
          icon="🔄"
          testId="stats-running"
        />
        <StatsCard
          title="Successful"
          value={stats?.success_count ?? 0}
          loading={statsLoading}
          variant="success"
          icon="✓"
          testId="stats-success"
        />
        <StatsCard
          title="Failed"
          value={stats?.failed_count ?? 0}
          loading={statsLoading}
          variant="failed"
          icon="✗"
          testId="stats-failed"
        />
      </div>

      {/* Status Distribution */}
      {stats && stats.by_result_status && stats.by_result_status.length > 0 && (
        <Card className="mb-8">
          <CardHeader className="pb-3">
            <CardTitle>Result Status Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-3">
              {stats.by_result_status.map((item) => (
                <div
                  key={item.status}
                  className="flex items-center gap-2 rounded-lg border bg-muted/40 px-3 py-2"
                >
                  <StatusBadge status={item.status} type="result" />
                  <span className="font-mono text-sm">{item.count}</span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Recent Experiments */}
      <Card className="overflow-hidden">
        <CardHeader className="flex-row items-center justify-between space-y-0 border-b">
          <CardTitle>Recent Experiments</CardTitle>
          <Button asChild variant="ghost" size="sm">
            <Link to="/experiments">View all →</Link>
          </Button>
        </CardHeader>
        {experimentsLoading ? (
          <div className="p-6 text-muted-foreground">Loading...</div>
        ) : recentExperiments?.experiments.length === 0 ? (
          <div className="p-6 text-muted-foreground">No experiments yet</div>
        ) : (
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="pl-6">Class</TableHead>
                <TableHead>Namespace</TableHead>
                <TableHead>Result</TableHead>
                <TableHead>Attempt</TableHead>
                <TableHead>Updated</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {recentExperiments?.experiments.map((exp) => (
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
                  <TableCell className="font-mono text-sm text-muted-foreground">
                    {exp.namespace}
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
                  <TableCell className="text-sm text-muted-foreground">
                    {exp.updated_at
                      ? new Date(exp.updated_at).toLocaleString()
                      : "—"}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Card>
    </div>
  );
}


