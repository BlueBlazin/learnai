import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectSeparator,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const ANY = "__any__";

/** A single-choice filter with an explicit "any" option, for example "Any judge". */
export function FilterSelect<T extends string>({
  label,
  anyLabel,
  value,
  options,
  onChange,
}: {
  /** Accessible name, for example "Judge". */
  label: string;
  anyLabel: string;
  value: T | null;
  options: { value: T; label: string }[];
  onChange: (value: T | null) => void;
}) {
  const items = [{ value: ANY, label: anyLabel }, ...options];
  return (
    <Select
      items={items}
      value={value ?? ANY}
      onValueChange={(next) => onChange(next === ANY || next === null ? null : (next as T))}
    >
      <SelectTrigger
        size="sm"
        aria-label={label}
        data-active={value !== null || undefined}
        className="text-muted-foreground data-active:text-foreground"
      >
        <SelectValue />
      </SelectTrigger>
      <SelectContent align="start" alignItemWithTrigger={false} className="min-w-44">
        <SelectGroup>
          <SelectItem value={ANY}>{anyLabel}</SelectItem>
        </SelectGroup>
        <SelectSeparator />
        <SelectGroup>
          {options.map((option) => (
            <SelectItem key={option.value} value={option.value}>
              {option.label}
            </SelectItem>
          ))}
        </SelectGroup>
      </SelectContent>
    </Select>
  );
}
