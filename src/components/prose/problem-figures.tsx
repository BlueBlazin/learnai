import { createContext, type ReactNode, useContext } from "react";
import { Notice } from "@/components/app/notice";
import { LoadingLines } from "@/components/app/states";
import { useProblemAsset } from "@/lib/api/queries";
import { AssetFigureContext, type AssetFigureProps } from "./asset-figure-context";
import { Figure, SvgFigureBody } from "./figure";
import { svgFromDataUrl } from "./svg-sanitize";

const ProblemIdContext = createContext<string | null>(null);

/**
 * A figure whose drawing is a file in a problem's `assets/` directory, fetched
 * through the backend as a `data:` URL: SVG is sanitized and inlined, PNG is an
 * image. While loading or when the file cannot be served, the numbered caption
 * still stands, so the text around it keeps its references.
 */
function ProblemAssetFigure({ path, number, alt, caption }: AssetFigureProps) {
  const asset = useProblemAsset(useContext(ProblemIdContext), path);
  const svg = asset.data ? svgFromDataUrl(asset.data) : null;
  return (
    <Figure number={number} caption={caption}>
      {asset.error ? (
        <Notice>The figure could not be loaded: {asset.error.message}</Notice>
      ) : !asset.data ? (
        <LoadingLines lines={3} />
      ) : svg !== null ? (
        <SvgFigureBody source={svg} label={alt} />
      ) : (
        <img src={asset.data} alt={alt} />
      )}
    </Figure>
  );
}

/**
 * Lets the prose inside it show a problem's figures: `![caption](assets/x.svg)`
 * in a statement resolves against that problem's `assets/` directory.
 */
export function ProblemFigures({ problemId, children }: { problemId: string; children: ReactNode }) {
  return (
    <ProblemIdContext value={problemId}>
      <AssetFigureContext value={ProblemAssetFigure}>{children}</AssetFigureContext>
    </ProblemIdContext>
  );
}
