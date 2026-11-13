import { cn } from "cn";
import type { ComponentProps } from "react";
import { Button } from "@/components/ui/button";

/**
 * A button set as text inside a sentence: it takes the type of the words
 * around it and wraps with them. The base for in-app links (`AppLink`), links
 * in prose, and terms that carry a tooltip.
 */
export function InlineButton({ className, ...props }: ComponentProps<typeof Button>) {
  return (
    <Button
      variant="link"
      className={cn("type-inherit h-auto rounded-sm p-0 whitespace-normal", className)}
      {...props}
    />
  );
}
