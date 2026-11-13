import { QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import React from "react";
import ReactDOM from "react-dom/client";
import { queryClient } from "@/lib/api/queries";
import { installNativeBehavior } from "@/lib/platform";
import { router } from "@/router";
import { installTheme } from "@/stores/theme";
import "./index.css";

installTheme();
installNativeBehavior();

ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>,
);
