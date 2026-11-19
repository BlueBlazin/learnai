/** A rectangle in window coordinates. */
export interface Box {
  top: number;
  left: number;
  bottom: number;
  right: number;
}

export interface InfoPlacement {
  /** The edge of the suggestion list that `offset` is measured from. */
  side: "top" | "bottom";
  offset: number;
  maxWidth: number;
  maxHeight: number;
  where: "left" | "right" | "right-narrow";
}

/** The widest the documentation panel gets, whatever room there is. */
const MAX_WIDTH = 400;
/** In the narrow layout the panel stops this far short of the window's edge. */
const NARROW_MARGIN = 30;

/**
 * Where the documentation panel of the suggestion list goes.
 *
 * CodeMirror's own placement starts the panel at the selected row and, when the
 * panel is taller than the room under that row, slides it up past the top of the
 * list, over the line being typed. Here the panel stays inside the band the list
 * occupies, on the far side of the cursor line: it starts level with the list's
 * top when the list hangs below the cursor (level with its bottom when the list
 * sits `above`) and is no taller than the room from there to the window's edge,
 * so a long entry scrolls inside the panel instead of covering the code.
 *
 * With no room on either side the panel goes under or over the selected row, as
 * CodeMirror does, limited the same way.
 */
export function placeInfo(input: {
  list: Box;
  option: Box;
  info: Box;
  space: Box;
  above: boolean;
}): InfoPlacement {
  const { list, option, info, space, above } = input;
  const spaceLeft = list.left - space.left;
  const spaceRight = space.right - list.right;
  const infoWidth = info.right - info.left;
  const infoHeight = info.bottom - info.top;
  const left = spaceRight < Math.min(infoWidth, spaceLeft);
  const beside = left ? spaceLeft : spaceRight;
  if (infoWidth <= beside) {
    return {
      side: above ? "bottom" : "top",
      offset: 0,
      maxWidth: Math.min(MAX_WIDTH, beside),
      maxHeight: Math.max(above ? list.bottom - space.top : space.bottom - list.top, 0),
      where: left ? "left" : "right",
    };
  }
  const maxWidth = Math.min(MAX_WIDTH, space.right - list.left - NARROW_MARGIN);
  const spaceBelow = space.bottom - list.bottom;
  if (spaceBelow >= infoHeight || spaceBelow > list.top - space.top) {
    return {
      side: "top",
      offset: option.bottom - list.top,
      maxWidth,
      maxHeight: Math.max(space.bottom - option.bottom, 0),
      where: "right-narrow",
    };
  }
  return {
    side: "bottom",
    offset: list.bottom - option.top,
    maxWidth,
    maxHeight: Math.max(option.top - space.top, 0),
    where: "right-narrow",
  };
}
