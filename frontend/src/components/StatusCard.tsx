import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface StatusCardProps {
  status: string;
  mode: string;
  // errors: string[]; // To be added later
}

const StatusIndicator = ({ status }: { status: string }) => {
  const bgColor =
    status === "running"
      ? "bg-green-500"
      : status === "stopped"
      ? "bg-gray-500"
      : "bg-red-500";
  return (
    <div className="flex items-center space-x-2">
      <span className={`h-3 w-3 rounded-full ${bgColor}`} />
      <span className="capitalize">{status}</span>
    </div>
  );
};

export function StatusCard({ status, mode }: StatusCardProps) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Status</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">State</span>
          <StatusIndicator status={status} />
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Mode</span>
          <span className="px-2 py-1 text-xs font-semibold rounded-full bg-secondary text-secondary-foreground">
            {mode.toUpperCase()}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
