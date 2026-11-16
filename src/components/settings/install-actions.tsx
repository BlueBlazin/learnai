import { LoaderCircle } from "lucide-react";
import type { ComponentProps } from "react";
import { Button } from "@/components/ui/button";
import { UV_INSTALL_URL } from "@/lib/environment";
import { openExternal } from "@/lib/platform";

/** Starts an install (a framework, or the base environment) and shows that it is running. */
export function InstallButton({
  installing,
  children,
  size = "xs",
  ...props
}: Omit<ComponentProps<typeof Button>, "variant"> & {
  /** True while this button's install is the one in flight. */
  installing: boolean;
}) {
  return (
    <Button variant="outline" size={size} {...props}>
      {installing ? <LoaderCircle className="animate-spin" data-icon="inline-start" /> : null}
      {installing ? "Installing" : children}
    </Button>
  );
}

/** Opens uv's installation guide in the system browser. Learn AI cannot install uv itself. */
export function UvGuideButton() {
  return (
    <Button variant="outline" size="xs" onClick={() => openExternal(UV_INSTALL_URL)}>
      How to install uv
    </Button>
  );
}
