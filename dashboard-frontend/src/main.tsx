import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { AxiosError } from "axios";

import { routeTree } from "./routeTree.gen";
import "./index.css";

// Create a query client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchInterval: 5000, // Poll every 5 seconds
      staleTime: 2000,
      // Retry on server errors (5xx) and network errors, but not client errors (4xx)
      // 4xx errors are deterministic - retrying won't help
      retry: (failureCount, error) => {
        if (error instanceof AxiosError && error.response?.status) {
          const status = error.response.status;
          // Don't retry client errors (400-499)
          if (status >= 400 && status < 500) {
            return false;
          }
        }
        // Retry server errors and network errors up to 3 times
        return failureCount < 3;
      },
    },
  },
});

// Create the router
const router = createRouter({ routeTree });

// Register types for type safety
declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>
);



