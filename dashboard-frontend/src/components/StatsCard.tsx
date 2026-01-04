interface StatsCardProps {
  title: string;
  value: number;
  loading?: boolean;
  variant?: "default" | "success" | "failed" | "running";
  icon?: string;
}

const variantStyles: Record<string, string> = {
  default: "border-slate-700",
  success: "border-emerald-700",
  failed: "border-red-700",
  running: "border-blue-700",
};

const valueStyles: Record<string, string> = {
  default: "text-white",
  success: "text-emerald-400",
  failed: "text-red-400",
  running: "text-blue-400",
};

export function StatsCard({
  title,
  value,
  loading,
  variant = "default",
  icon,
}: StatsCardProps) {
  return (
    <div
      className={`bg-slate-900 border rounded-lg p-5 ${variantStyles[variant]}`}
    >
      <div className="flex items-center justify-between mb-2">
        <span className="text-slate-400 text-sm">{title}</span>
        {icon && <span className="text-lg opacity-50">{icon}</span>}
      </div>
      <div className={`text-3xl font-bold font-mono ${valueStyles[variant]}`}>
        {loading ? "..." : value.toLocaleString()}
      </div>
    </div>
  );
}



