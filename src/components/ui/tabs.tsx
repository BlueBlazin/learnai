import { Tabs as TabsPrimitive } from "@base-ui/react/tabs"
import { cn } from "cn"

/**
 * Tabs in the Whetstone system are plain text with a rule under the active one;
 * there is no pill or segmented variant.
 */
function Tabs({ className, ...props }: TabsPrimitive.Root.Props) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      className={cn("flex flex-col", className)}
      {...props}
    />
  )
}

function TabsList({ className, ...props }: TabsPrimitive.List.Props) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      className={cn("relative flex items-stretch gap-5 text-muted-foreground", className)}
      {...props}
    />
  )
}

function TabsTrigger({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-trigger"
      className={cn(
        "relative inline-flex items-center gap-1.5 text-sm font-medium whitespace-nowrap text-muted-foreground transition-colors outline-none select-none hover:text-foreground focus-visible:text-foreground focus-visible:underline focus-visible:underline-offset-4 disabled:pointer-events-none disabled:opacity-50 data-active:text-foreground",
        "after:absolute after:inset-x-0 after:-bottom-px after:h-0.5 after:bg-foreground after:opacity-0 data-active:after:opacity-100",
        "[&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-3.5",
        className
      )}
      {...props}
    />
  )
}

function TabsContent({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-content"
      className={cn("min-h-0 flex-1 outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent }
