import { Card, CardContent } from "./ui/card";

interface EmptyStateProps {
  title: string;
  description?: string;
  icon?: string;
}

export function EmptyState({
  title,
  description,
  icon = "📭",
}: EmptyStateProps) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex flex-col items-center justify-center py-16 text-center">
        <span className="mb-4 text-5xl">{icon}</span>
        <h3 className="mb-2 text-lg font-medium">{title}</h3>
        {description ? (
          <p className="max-w-md text-muted-foreground">{description}</p>
        ) : null}
      </CardContent>
    </Card>
  );
}


