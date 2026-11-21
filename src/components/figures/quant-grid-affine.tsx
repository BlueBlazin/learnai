import QuantGrid from "./quant-grid";
import type { InteractiveFigureProps } from "./registry";

/** `quant-grid-affine`: the same figure, opening on the affine grid of problems/quantization/affine-int8 at 8 bits. */
export default function QuantGridAffine(props: InteractiveFigureProps) {
  return <QuantGrid {...props} initialMode="affine" initialBits={8} />;
}
