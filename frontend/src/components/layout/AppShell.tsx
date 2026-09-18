import { Outlet } from "react-router-dom";
import { AppHeader } from "./AppHeader";
import { Sidebar } from "./Sidebar";

export function AppShell() {
  return (
    <div className="flex h-screen flex-col overflow-hidden bg-canvas">
      <AppHeader />
      <div className="relative flex flex-1 overflow-hidden">
        <div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            background:
              "radial-gradient(38% 30% at 100% 0%, rgba(209,33,38,0.06), transparent 70%)," +
              "radial-gradient(42% 34% at 100% 100%, rgba(209,33,38,0.055), transparent 70%)",
          }}
        />
        <Sidebar />
        <main className="relative flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1360px] px-8 py-8">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  );
}
