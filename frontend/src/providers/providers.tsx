"use client";

import { QueryProvider } from "./query-provider";
import { AuthProvider } from "./auth-provider";
import { RequestActivityIndicator } from "@/components/shared/request-activity-indicator";

export function Providers({ children }: { children: React.ReactNode }) {
  return (
    <QueryProvider>
      <AuthProvider>
        {children}
        <RequestActivityIndicator />
      </AuthProvider>
    </QueryProvider>
  );
}
