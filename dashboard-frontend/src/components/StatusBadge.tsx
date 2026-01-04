import type * as React from "react";
import { Badge } from "./ui/badge";

interface StatusBadgeProps {
  status: string;
  type: "result" | "attempt";
}

const resultStatusVariants: Record<string, React.ComponentProps<typeof Badge>["variant"]> = {
  success: "success",
  failed: "destructive",
  incomplete: "warning",
  absent: "muted",
};

const attemptStatusVariants: Record<string, React.ComponentProps<typeof Badge>["variant"]> = {
  success: "success",
  running: "info",
  queued: "secondary",
  failed: "destructive",
  crashed: "warning",
  cancelled: "muted",
  preempted: "warning",
};

export function StatusBadge({ status, type }: StatusBadgeProps) {
  const variantMap =
    type === "result" ? resultStatusVariants : attemptStatusVariants;
  const variant = variantMap[status] ?? "muted";

  return (
    <Badge variant={variant} className="gap-1.5 px-2.5 py-1">
      {type === "attempt" && status === "running" && (
        <span className="h-1.5 w-1.5 rounded-full bg-blue-300 animate-pulse" />
      )}
      {status}
    </Badge>
  );
}

