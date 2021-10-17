import { Results as HandResults } from '@mediapipe/hands';
import { Results as FaceResults } from '@mediapipe/face_mesh';
import { Results as PoseResults } from '@mediapipe/pose';

export type ResultsType = HandResults | FaceResults | PoseResults;

export function isHandsResults(results: ResultsType): results is HandResults {
  return true;
}

export function isFaceResults(results: ResultsType): results is FaceResults {
  return true;
}

export function isPoseResults(results: ResultsType): results is PoseResults {
  return true;
}
