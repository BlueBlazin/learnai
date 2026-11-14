import { type ComponentType, createContext, type ReactNode } from "react";

export interface AssetFigureProps {
  /** Path of the file, relative to the `assets/` directory the text belongs to. */
  path: string;
  number: number;
  /** The caption as plain text: the drawing's accessible name. */
  alt: string;
  caption: ReactNode;
}

/**
 * How prose draws `![caption](assets/<file>)`. The Markdown renderer knows
 * nothing about problems or the backend: whoever knows where the text's assets
 * live supplies the component (`ProblemFigures`). Null: there are no assets.
 */
export const AssetFigureContext = createContext<ComponentType<AssetFigureProps> | null>(null);
