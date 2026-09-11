import { Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { PageHeader } from "./components/layout/PageHeader";
import { EmptyState } from "./components/ui/EmptyState";

function HomePage() {
  return (
    <>
      <PageHeader
        eyebrow="Adinn"
        title="4K Image Enhancer"
        subtitle="Upload one or more photos to restore, denoise, and upscale them to 4K."
      />
      <div className="mt-8">
        <EmptyState
          title="Upload flow coming soon"
          description="The new enhancement UI is under construction. This is the clean base for the Adinn 4K Image Enhancer product."
        />
      </div>
    </>
  );
}

export default function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<HomePage />} />
      </Route>
    </Routes>
  );
}
