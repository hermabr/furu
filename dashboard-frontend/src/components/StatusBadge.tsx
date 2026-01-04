interface StatusBadgeProps {
  status: string;
  type: "result" | "attempt";
}

const resultStatusColors: Record<string, string> = {
  success: "bg-emerald-900/50 text-emerald-400 border-emerald-700",
  failed: "bg-red-900/50 text-red-400 border-red-700",
  incomplete: "bg-amber-900/50 text-amber-400 border-amber-700",
  absent: "bg-slate-800 text-slate-400 border-slate-600",
};

const attemptStatusColors: Record<string, string> = {
  success: "bg-emerald-900/50 text-emerald-400 border-emerald-700",
  running: "bg-blue-900/50 text-blue-400 border-blue-700",
  queued: "bg-purple-900/50 text-purple-400 border-purple-700",
  failed: "bg-red-900/50 text-red-400 border-red-700",
  crashed: "bg-orange-900/50 text-orange-400 border-orange-700",
  cancelled: "bg-slate-800 text-slate-400 border-slate-600",
  preempted: "bg-yellow-900/50 text-yellow-400 border-yellow-700",
};

export function StatusBadge({ status, type }: StatusBadgeProps) {
  const colorMap = type === "result" ? resultStatusColors : attemptStatusColors;
  const colors = colorMap[status] || "bg-slate-800 text-slate-400 border-slate-600";

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${colors}`}
    >
      {type === "attempt" && status === "running" && (
        <span className="w-1.5 h-1.5 bg-blue-400 rounded-full animate-pulse"></span>
      )}
      {status}
    </span>
  );
}



