import { createRootRoute, Link, Outlet } from "@tanstack/react-router";

export const Route = createRootRoute({
  component: RootComponent,
});

function RootComponent() {
  return (
    <div className="min-h-screen flex flex-col bg-slate-950">
      <nav className="px-8 py-4 border-b border-slate-800 bg-slate-900 flex gap-6 items-center">
        <Link
          to="/"
          className="font-bold text-xl text-emerald-400 no-underline flex items-center gap-2"
        >
          <svg
            className="w-7 h-7"
            viewBox="0 0 32 32"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
          >
            <rect width="32" height="32" rx="6" fill="#166534" />
            <path
              d="M8 10h4v12H8V10zm6 0h4v12h-4V10zm6 0h4v12h-4V10z"
              fill="#86efac"
            />
            <circle cx="10" cy="13" r="1.5" fill="#22c55e" />
            <circle cx="16" cy="16" r="1.5" fill="#22c55e" />
            <circle cx="22" cy="19" r="1.5" fill="#22c55e" />
          </svg>
          Huldra
        </Link>
        <div className="flex gap-4">
          <Link
            to="/"
            className="text-slate-400 no-underline hover:text-emerald-400 transition-colors [&.active]:text-emerald-400 [&.active]:font-medium"
            activeOptions={{ exact: true }}
          >
            Dashboard
          </Link>
          <Link
            to="/experiments"
            className="text-slate-400 no-underline hover:text-emerald-400 transition-colors [&.active]:text-emerald-400 [&.active]:font-medium"
          >
            Experiments
          </Link>
        </div>
      </nav>
      <main className="flex-1 p-8">
        <Outlet />
      </main>
    </div>
  );
}



