import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Dashboard } from "@/components/Dashboard";
import { useWebSocket } from "@/hooks/useWebSocket";

interface TelemetryData {
  [branchName: string]: any;
}

function App() {
  const { latestMessage, isConnected } = useWebSocket();
  const [telemetryData, setTelemetryData] = useState<TelemetryData>({});

  useEffect(() => {
    if (latestMessage) {
      setTelemetryData((prevData) => ({
        ...prevData,
        [latestMessage.branch]: latestMessage,
      }));
    }
  }, [latestMessage]);

  return (
    <div className="dark bg-background text-foreground min-h-screen flex flex-col">
      <header className="p-4 border-b flex justify-between items-center">
        <h1 className="text-2xl font-bold">Trading Bot Control Plane</h1>
        <div className="flex items-center space-x-2">
          <span className={`h-3 w-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`}></span>
          <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </header>
      <div className="flex flex-1">
        <aside className="w-64 p-4 border-r flex flex-col space-y-4">
          <h2 className="text-lg font-semibold">Global Controls</h2>
          <Button>Start All</Button>
          <Button variant="destructive">Stop All</Button>
          <Button variant="outline">Restart All</Button>
          <div className="pt-4 border-t">
            <h3 className="text-md font-semibold">Branch Controls</h3>
            <p className="text-sm text-muted-foreground">(Select a tab to see branch-specific controls)</p>
          </div>
        </aside>
        <main className="flex-1 p-4">
          <Tabs defaultValue="crypto_spot" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="crypto_spot">Crypto Spot</TabsTrigger>
              <TabsTrigger value="crypto_futures">Crypto Futures</TabsTrigger>
              <TabsTrigger value="forex_spot">Forex Spot</TabsTrigger>
              <TabsTrigger value="forex_options">Forex Options</TabsTrigger>
            </TabsList>
            <TabsContent value="crypto_spot">
              <Dashboard branchName="CRYPTO_SPOT" telemetry={telemetryData["CRYPTO_SPOT"]} />
            </TabsContent>
            <TabsContent value="crypto_futures">
              <Dashboard branchName="CRYPTO_FUTURES" telemetry={telemetryData["CRYPTO_FUTURES"]} />
            </TabsContent>
            <TabsContent value="forex_spot">
              <Dashboard branchName="FOREX_SPOT" telemetry={telemetryData["FOREX_SPOT"]} />
            </TabsContent>
            <TabsContent value="forex_options">
              <Dashboard branchName="FOREX_OPTIONS" telemetry={telemetryData["FOREX_OPTIONS"]} />
            </TabsContent>
          </Tabs>
        </main>
      </div>
    </div>
  );
}

export default App;
