import { useEffect, useState } from "react";
import axios from "axios";
import { StatusCard } from "./StatusCard";
import { BalanceCard } from "./BalanceCard";

interface DashboardProps {
  branchName: string;
  telemetry: any;
}

interface StatusData {
  status: string;
  mode: string;
}

interface BalanceData {
  total: number;
  available: number;
  used: number;
  currency: string;
}

// A helper to process the raw balance data from the API
const processBalanceData = (rawData: any): BalanceData | null => {
    if (!rawData || !rawData.list || rawData.list.length === 0) {
        return { total: 0, available: 0, used: 0, currency: "USD" };
    }

    // For Bybit UNIFIED account, we look for USDT
    const account = rawData.list[0];
    const coin = account.coin.find((c: any) => c.coin === 'USDT');

    if (!coin) {
        return { total: 0, available: 0, used: 0, currency: "USDT" };
    }

    const total = parseFloat(coin.equity) || 0;
    const available = parseFloat(coin.availableToWithdraw) || 0;
    const used = total - available;

    return {
        total,
        available,
        used,
        currency: "USDT"
    };
};


export function Dashboard({ branchName, telemetry }: DashboardProps) {
  const [status, setStatus] = useState<StatusData>({ status: "loading...", mode: "..."});
  const [balance, setBalance] = useState<BalanceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Effect for handling incoming telemetry data for status
  useEffect(() => {
    if (telemetry) {
      setStatus({
        status: telemetry.status,
        mode: telemetry.mode,
      });
    }
  }, [telemetry]);

  // Effect for fetching less frequent data like balance
  useEffect(() => {
    const fetchBalance = async () => {
      try {
        const balanceRes = await axios.get(`/branches/${branchName}/balances`);
        const processedBalance = processBalanceData(balanceRes.data);
        setBalance(processedBalance);
        if (error) setError(null); // Clear previous errors on success
      } catch (err) {
        if (axios.isAxiosError(err)) {
          setError(err.response?.data?.detail || "Failed to fetch balance");
        } else {
          setError("An unknown error occurred while fetching balance");
        }
        console.error(`Error fetching balance for ${branchName}:`, err);
      }
    };

    fetchBalance();
    const interval = setInterval(fetchBalance, 15000); // Fetch balance every 15 seconds

    return () => clearInterval(interval);
  }, [branchName, error]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      <div className="lg:col-span-1">
        <StatusCard status={status.status} mode={status.mode} />
      </div>
      <div className="lg:col-span-1">
        <BalanceCard balance={balance} />
      </div>
      {error && <div className="lg:col-span-3 p-4 text-red-500 bg-red-900/20 rounded-md">Error: {error}</div>}
      {/* Other components like KPIs, Positions, etc. will go here */}
    </div>
  );
}
