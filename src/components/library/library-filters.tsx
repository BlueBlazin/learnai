import { Search, X } from "lucide-react";
import { DifficultyMark } from "@/components/problem/difficulty-mark";
import { Button } from "@/components/ui/button";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { DIFFICULTIES, type Difficulty, FRAMEWORKS, JUDGES, STATUSES } from "@/lib/api";
import { activeFilterCount, type LibraryFilters } from "@/lib/filters";
import { FRAMEWORK_LABEL, JUDGE_LABEL, STATUS_LABEL } from "@/lib/labels";
import { FilterSelect } from "./filter-select";

const STATUS_OPTIONS = STATUSES.map((value) => ({ value, label: STATUS_LABEL[value] }));
const FRAMEWORK_OPTIONS = FRAMEWORKS.map((value) => ({ value, label: FRAMEWORK_LABEL[value] }));
const JUDGE_OPTIONS = JUDGES.map((value) => ({ value, label: JUDGE_LABEL[value] }));

/** The library's filter bar: text search, difficulty, status, framework, judge. */
export function LibraryFilterBar({
  filters,
  onChange,
  searchRef,
}: {
  filters: LibraryFilters;
  onChange: (filters: LibraryFilters) => void;
  searchRef?: React.Ref<HTMLInputElement>;
}) {
  const patch = (changes: Partial<LibraryFilters>) => onChange({ ...filters, ...changes });
  // The track is chosen in the sidebar; "Clear" only resets what this bar controls.
  const active = activeFilterCount({ ...filters, track: null });

  return (
    <div className="flex flex-wrap items-center gap-2">
      <InputGroup className="h-7 w-56 min-w-40 flex-1">
        <InputGroupAddon>
          <Search aria-hidden />
        </InputGroupAddon>
        <InputGroupInput
          ref={searchRef}
          type="search"
          aria-label="Search problems"
          placeholder="Search by title or tag"
          value={filters.q}
          onChange={(event) => patch({ q: event.target.value })}
          spellCheck={false}
        />
        {filters.q ? (
          <InputGroupAddon align="inline-end">
            <InputGroupButton size="icon-xs" aria-label="Clear search" onClick={() => patch({ q: "" })}>
              <X />
            </InputGroupButton>
          </InputGroupAddon>
        ) : null}
      </InputGroup>

      <ToggleGroup
        multiple
        variant="outline"
        size="sm"
        spacing={0}
        aria-label="Difficulty"
        value={filters.difficulty}
        onValueChange={(value) =>
          patch({ difficulty: DIFFICULTIES.filter((d) => (value as Difficulty[]).includes(d)) })
        }
      >
        {DIFFICULTIES.map((difficulty) => (
          <ToggleGroupItem key={difficulty} value={difficulty} className="text-xs">
            <DifficultyMark difficulty={difficulty} />
          </ToggleGroupItem>
        ))}
      </ToggleGroup>

      <FilterSelect
        label="Status"
        anyLabel="Any status"
        value={filters.status[0] ?? null}
        options={STATUS_OPTIONS}
        onChange={(status) => patch({ status: status ? [status] : [] })}
      />
      <FilterSelect
        label="Framework"
        anyLabel="Any framework"
        value={filters.framework}
        options={FRAMEWORK_OPTIONS}
        onChange={(framework) => patch({ framework })}
      />
      <FilterSelect
        label="Judge"
        anyLabel="Any judge"
        value={filters.judge}
        options={JUDGE_OPTIONS}
        onChange={(judge) => patch({ judge })}
      />
      {active > 0 ? (
        <Button
          variant="ghost"
          size="sm"
          className="text-muted-foreground"
          onClick={() =>
            onChange({ ...filters, q: "", difficulty: [], status: [], framework: null, judge: null })
          }
        >
          Clear filters
        </Button>
      ) : null}
    </div>
  );
}
