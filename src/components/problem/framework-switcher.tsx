import { Check } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { Framework, FrameworkEnv } from "@/lib/api";
import { FRAMEWORK_LABEL } from "@/lib/labels";

/**
 * Chooses the framework a solution is written in. Each framework has its own
 * solution file. Shows which are already solved and which are not installed.
 */
export function FrameworkSwitcher({
  frameworks,
  value,
  onChange,
  solved = [],
  env,
  disabled = false,
}: {
  frameworks: Framework[];
  value: Framework;
  onChange: (framework: Framework) => void;
  solved?: Framework[];
  /** Environment status, when known, to flag frameworks that are not installed. */
  env?: FrameworkEnv[];
  disabled?: boolean;
}) {
  if (frameworks.length <= 1) {
    return <span className="px-1 text-sm font-medium">{FRAMEWORK_LABEL[value]}</span>;
  }
  const missing = (framework: Framework) =>
    env?.some((entry) => entry.framework === framework && !entry.installed) ?? false;
  const items = frameworks.map((framework) => ({ value: framework, label: FRAMEWORK_LABEL[framework] }));

  return (
    <Select
      items={items}
      value={value}
      onValueChange={(next) => next && onChange(next as Framework)}
      disabled={disabled}
    >
      <SelectTrigger size="sm" aria-label="Framework" className="min-w-28 font-medium">
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="start" alignItemWithTrigger={false} className="min-w-52">
        <SelectGroup>
          {frameworks.map((framework) => (
            <SelectItem key={framework} value={framework}>
              <span>{FRAMEWORK_LABEL[framework]}</span>
              {solved.includes(framework) ? (
                <span className="flex items-center gap-1 text-xs text-pass">
                  <Check className="size-3" aria-hidden />
                  solved
                </span>
              ) : null}
              {missing(framework) ? (
                <span className="text-xs text-muted-foreground">not installed</span>
              ) : null}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
