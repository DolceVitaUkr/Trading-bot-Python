import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface Balance {
  total: number;
  available: number;
  used: number;
  currency: string;
}

interface BalanceCardProps {
  balance: Balance | null;
}

const formatCurrency = (amount: number, currency: string) => {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(amount);
};

export function BalanceCard({ balance }: BalanceCardProps) {
  const currency = balance?.currency || "USD";

  return (
    <Card>
      <CardHeader>
        <CardTitle>Balance</CardTitle>
      </CardHeader>
      <CardContent className="grid gap-4">
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Available</span>
          <span className="font-bold">
            {balance ? formatCurrency(balance.available, currency) : "Loading..."}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-muted-foreground">Used / In Orders</span>
          <span className="font-bold">
            {balance ? formatCurrency(balance.used, currency) : "Loading..."}
          </span>
        </div>
        <div className="flex items-center justify-between border-t pt-4 mt-2">
          <span className="text-lg font-semibold">Total</span>
          <span className="text-lg font-bold">
            {balance ? formatCurrency(balance.total, currency) : "Loading..."}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}
