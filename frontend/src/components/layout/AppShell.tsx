import { Outlet } from "react-router-dom";
import { AppHeader } from "./AppHeader";

export function AppShell() {
  return (
    <div className="flex min-h-screen flex-col">
      <AppHeader />
      <main className="flex-1">
        <div className="mx-auto w-full max-w-[1180px] px-6 py-8">
          <Outlet />
        </div>
      </main>
      <footer className="border-t border-line bg-white">
        <div className="mx-auto flex max-w-[1180px] items-center justify-between px-6 py-4">
          <p className="text-[11.5px] text-faint">Adinn — 4K Image Enhancer</p>
        </div>
      </footer>
    </div>
  );
}
