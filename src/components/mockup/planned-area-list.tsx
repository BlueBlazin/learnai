import { RuledList } from "@/components/app/ruled-list";

/**
 * Areas that are planned but not written: a title, a sentence, and the word
 * "Planned". No art and nothing to open, so they stay secondary to the areas
 * that exist without looking broken.
 */
export function PlannedAreaList({ areas }: { areas: readonly { title: string; description: string }[] }) {
  return (
    <RuledList framed={false} aria-label="Planned areas">
      {areas.map((area) => (
        <li key={area.title} className="flex items-baseline gap-8 py-4">
          <h3 className="w-52 shrink-0 font-serif text-lg">{area.title}</h3>
          <p className="min-w-0 flex-1 text-sm text-pretty text-muted-foreground">{area.description}</p>
          <span className="shrink-0 text-xs text-muted-foreground">Planned</span>
        </li>
      ))}
    </RuledList>
  );
}
