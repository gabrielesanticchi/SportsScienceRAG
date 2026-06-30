import * as React from "react";

import { cn } from "@/lib/utils";

export interface BadgeProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: "default" | "secondary" | "success" | "warning" | "destructive";
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <div
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        {
          default: "bg-zinc-900 text-white",
          secondary: "bg-zinc-100 text-zinc-700",
          success: "bg-emerald-100 text-emerald-800",
          warning: "bg-amber-100 text-amber-800",
          destructive: "bg-red-100 text-red-800",
        }[variant],
        className,
      )}
      {...props}
    />
  );
}

export { Badge };
