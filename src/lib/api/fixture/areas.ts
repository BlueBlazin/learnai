import { FIXTURE_TRACKS } from "./tracks";

/** A fixture area before validation: `modules` lists fixture track ids. */
export interface FixtureArea {
  id: string;
  title: string;
  description: string;
  modules: string[];
}

/** The fixture's one area: every fixture track, in catalog order. */
export const FIXTURE_AREAS: FixtureArea[] = [
  {
    id: "inference-and-serving",
    title: "Inference and serving",
    description:
      "Everything between trained weights and a served answer: tokenizing, the forward pass, sampling, caching, quantization, and serving.",
    modules: FIXTURE_TRACKS.map((track) => track.id),
  },
];
