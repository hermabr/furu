import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader } from "./ui/card";
import { cn } from "../lib/utils";

interface StatsCardProps {
  title: string;
  value: number;
  loading?: boolean;
  variant?: "default" | "success" | "failed" | "running";
  icon?: string;
  testId?: string;
}

const borderStyles: Record<string, string> = {
  default: "border-border",
  success: "border-emerald-500/40",
  failed: "border-red-500/40",
  running: "border-blue-500/40",
};

const valueStyles: Record<string, string> = {
  default: "text-foreground",
  success: "text-emerald-300",
  failed: "text-red-300",
  running: "text-blue-300",
};

export function StatsCard({
  title,
  value,
  loading,
  variant = "default",
  icon,
  testId,
}: StatsCardProps) {
  return (
    <Card className={cn("p-5", borderStyles[variant])} data-testid={testId}>
      <CardHeader className="p-0 pb-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-muted-foreground">{title}</span>
          {icon ? (
            <Badge variant="secondary" className="text-base px-2 py-1">
              {icon}
            </Badge>
          ) : null}
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div
          className={cn(
            "text-3xl font-bold font-mono",
            valueStyles[variant],
          )}
          data-testid={testId ? `${testId}-value` : undefined}
        >
          {loading ? "..." : value.toLocaleString()}
        </div>
      </CardContent>
    </Card>
  );
}


