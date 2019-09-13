package com.indigoviolet.posenet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

public class PosenetDecoder {

    private String[] partNames = {
        "nose", "leftEye", "rightEye", "leftEar", "rightEar", "leftShoulder",
        "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist",
        "leftHip", "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"
    };

    private String[][] poseChain = {
        {"nose", "leftEye"}, {"leftEye", "leftEar"}, {"nose", "rightEye"},
        {"rightEye", "rightEar"}, {"nose", "leftShoulder"},
        {"leftShoulder", "leftElbow"}, {"leftElbow", "leftWrist"},
        {"leftShoulder", "leftHip"}, {"leftHip", "leftKnee"},
        {"leftKnee", "leftAnkle"}, {"nose", "rightShoulder"},
        {"rightShoulder", "rightElbow"}, {"rightElbow", "rightWrist"},
        {"rightShoulder", "rightHip"}, {"rightHip", "rightKnee"},
        {"rightKnee", "rightAnkle"}
    };

    private int mOutputStride;
    private int mNumKeypoints;

    public PosenetDecoder(int outputStride) {
        mOutputStride = outputStride;
        initPoseNet();
    }

    private Map<String, Integer> partsIds = new HashMap<>();
    private List<Integer> parentToChildEdges = new ArrayList<>();
    private List<Integer> childToParentEdges = new ArrayList<>();

    private void initPoseNet() {
        mNumKeypoints = partNames.length;
        for (int i = 0; i < mNumKeypoints; ++i)
            partsIds.put(partNames[i], i);

        for (String[] edge : poseChain) {
            parentToChildEdges.add(partsIds.get(edge[1]));
            childToParentEdges.add(partsIds.get(edge[0]));
        }
    }

    private Map<String, Object> makeKeypoint(float score, int partId, float y, float x) {
        Map<String, Object> keypoint = new HashMap<>();
        keypoint.put("score", score);
        keypoint.put("partId", partId);
        keypoint.put("part", partNames[partId]);

        Map<String, Float> position = new HashMap<>();
        position.put("y", y);
        position.put("x", x);
        keypoint.put("position", position);

        // keypoint.put("y", y);
        // keypoint.put("x", x);
        return keypoint;
    }

    public List<Map<String, Object>> decode(Map<Integer, Object> outputMap, final int numResults, final float threshold, final int nmsRadius) {
        int localMaximumRadius = 1;

        float[][][] scores = ((float[][][][]) outputMap.get(0))[0];
        float[][][] offsets = ((float[][][][]) outputMap.get(1))[0];
        float[][][] displacementsFwd = ((float[][][][]) outputMap.get(2))[0];
        float[][][] displacementsBwd = ((float[][][][]) outputMap.get(3))[0];

        PriorityQueue<Map<String, Object>> pq = buildPartWithScoreQueue(scores, threshold, localMaximumRadius);

        int squaredNmsRadius = nmsRadius * nmsRadius;

        List<Map<String, Object>> poses = new ArrayList<>();
        while (poses.size() < numResults && pq.size() > 0) {
            Map<String, Object> root = pq.poll();
            float[] rootPoint = getImageCoords(root, offsets);

            if (withinNmsRadiusOfCorrespondingPoint(poses, squaredNmsRadius, rootPoint[0], rootPoint[1], (int) root.get("partId")))
                continue;

            List<Map<String, Object>> keypoints = decodePose(root, rootPoint, scores, offsets, displacementsFwd, displacementsBwd);
            float score = getInstanceScore(keypoints, poses, squaredNmsRadius);

            Map<String, Object> pose = new HashMap<>();
            pose.put("keypoints", keypoints);
            pose.put("score", score);
            poses.add(pose);
        }
        return poses;
    }

    private List<Map<String, Object>> decodePose(Map<String, Object> root, float[] rootPoint, float[][][] scores, float[][][] offsets, float[][][] displacementsFwd, float[][][] displacementsBwd) {
        int numEdges = parentToChildEdges.size();
        Map<String, Object> rootKeypoint = makeKeypoint((float) root.get("score"), (int) root.get("partId"), rootPoint[0], rootPoint[1]);

        Map<Integer, Map<String, Object>> keypoints = new HashMap<>();
        keypoints.put((int) root.get("partId"), rootKeypoint);

        for (int edge = numEdges - 1; edge >= 0; --edge) {
            int sourceKeypointId = parentToChildEdges.get(edge);
            int targetKeypointId = childToParentEdges.get(edge);
            if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                Map<String, Object> keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId), targetKeypointId, scores, offsets, displacementsBwd);
                keypoints.put(targetKeypointId, keypoint);
            }
        }

        for (int edge = 0; edge < numEdges; ++edge) {
            int sourceKeypointId = childToParentEdges.get(edge);
            int targetKeypointId = parentToChildEdges.get(edge);
            if (keypoints.containsKey(sourceKeypointId) && !keypoints.containsKey(targetKeypointId)) {
                Map<String, Object> keypoint = traverseToTargetKeypoint(edge, keypoints.get(sourceKeypointId), targetKeypointId, scores, offsets,  displacementsFwd);
                keypoints.put(targetKeypointId, keypoint);
            }
        }
        return new ArrayList<>(keypoints.values());
    }

    private PriorityQueue<Map<String, Object>> buildPartWithScoreQueue(float[][][] scores, double threshold, int localMaximumRadius) {
        PriorityQueue<Map<String, Object>> pq =
            new PriorityQueue<>(
                                1,
                                new Comparator<Map<String, Object>>() {
                                    @Override
                                    public int compare(Map<String, Object> lhs, Map<String, Object> rhs) {
                                        return Float.compare((float) rhs.get("score"), (float) lhs.get("score"));
                                    }
                                });

        for (int heatmapY = 0; heatmapY < scores.length; ++heatmapY) {
            for (int heatmapX = 0; heatmapX < scores[0].length; ++heatmapX) {
                for (int keypointId = 0; keypointId < scores[0][0].length; ++keypointId) {
                    float score = sigmoid(scores[heatmapY][heatmapX][keypointId]);
                    if (score < threshold) continue;

                    if (scoreIsMaximumInLocalWindow(keypointId, score, heatmapY, heatmapX, localMaximumRadius, scores)) {
                        Map<String, Object> res = makeKeypoint(score, keypointId, heatmapY, heatmapX);
                        pq.add(res);
                    }
                }
            }
        }

        return pq;
    }

    private boolean scoreIsMaximumInLocalWindow(int keypointId,
                                                float score,
                                                int heatmapY,
                                                int heatmapX,
                                                int localMaximumRadius,
                                                float[][][] scores) {
        boolean localMaximum = true;
        int height = scores.length;
        int width = scores[0].length;

        int yStart = Math.max(heatmapY - localMaximumRadius, 0);
        int yEnd = Math.min(heatmapY + localMaximumRadius + 1, height);
        for (int yCurrent = yStart; yCurrent < yEnd; ++yCurrent) {
            int xStart = Math.max(heatmapX - localMaximumRadius, 0);
            int xEnd = Math.min(heatmapX + localMaximumRadius + 1, width);
            for (int xCurrent = xStart; xCurrent < xEnd; ++xCurrent) {
                if (sigmoid(scores[yCurrent][xCurrent][keypointId]) > score) {
                    localMaximum = false;
                    break;
                }
            }
            if (!localMaximum) {
                break;
            }
        }

        return localMaximum;
    }

    private float[] getOffsetPoint(int y, int x, int keypointId, float[][][] offsets) {
        float offsetY = offsets[y][x][keypointId];
        float offsetX = offsets[y][x][keypointId + mNumKeypoints];
        return new float[]{offsetY, offsetX};
    }

    private float[] getImageCoords(Map<String, Object> keypoint, float[][][] offsets) {
        // This is only invoked from keypoints that are on the
        // PriorityQueue, where x and y are integers, so it's safe to
        // round them (we can't cast float to int)
        int heatmapY = Math.round(getKeypointPosition(keypoint, "y"));
        int heatmapX = Math.round(getKeypointPosition(keypoint, "x"));
        int keypointId = (int) keypoint.get("partId");
        float[] offsetPoint = getOffsetPoint(heatmapY, heatmapX, keypointId, offsets);
        // int keypointId = (int) keypoint.get("partId");
        // float offsetY = offsets[heatmapY][heatmapX][keypointId];
        // float offsetX = offsets[heatmapY][heatmapX][keypointId + numParts];

        float y = heatmapY * mOutputStride + offsetPoint[0];
        float x = heatmapX * mOutputStride + offsetPoint[1];

        return new float[]{y, x};
    }

    private float squaredDistance(float y1, float x1, float y2, float x2) {
        float dy = y1 - y2;
        float dx = x1 - x2;
        return dy * dy + dx * dx;
    }

    private boolean withinNmsRadiusOfCorrespondingPoint(List<Map<String, Object>> poses,
                                                        float squaredNmsRadius,
                                                        float y,
                                                        float x,
                                                        int keypointId) {
        for (Map<String, Object> pose : poses) {
            List<Map<String, Object>> keypoints = (List<Map<String, Object>>) pose.get("keypoints");
            Map<String, Object> correspondingKeypoint = keypoints.get(keypointId);
            float sq = squaredDistance(y, x, getKeypointPosition(correspondingKeypoint, "y"),  getKeypointPosition(correspondingKeypoint, "x"));
            if (sq <= squaredNmsRadius)
                return true;
        }

        return false;
    }

    private float getKeypointPosition(Map<String, Object> keypoint, String axis) {
        return ((Map<String, Float>) keypoint.get("position")).get(axis);
    }



    private Map<String, Object> traverseToTargetKeypoint(int edgeId,
                                                         Map<String, Object> sourceKeypoint,
                                                         int targetKeypointId,
                                                         float[][][] scores,
                                                         float[][][] offsets,
                                                         float[][][] displacements) {
        int height = scores.length;
        int width = scores[0].length;
        float sourceKeypointY = getKeypointPosition(sourceKeypoint, "y");
        float sourceKeypointX = getKeypointPosition(sourceKeypoint, "x");

        int[] sourceKeypointIndices = getStridedIndexNearPoint(sourceKeypointY, sourceKeypointX, height, width);

        float[] displacement = getDisplacement(edgeId, sourceKeypointIndices, displacements);

        float[] displacedPoint = new float[]{
            sourceKeypointY + displacement[0],
            sourceKeypointX + displacement[1]
        };

        float[] targetKeypoint = displacedPoint;

        final int offsetRefineStep = 2;
        for (int i = 0; i < offsetRefineStep; i++) {
            int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1], height, width);

            int targetKeypointY = targetKeypointIndices[0];
            int targetKeypointX = targetKeypointIndices[1];

            float[] offsetPoint = getOffsetPoint(targetKeypointY, targetKeypointX, targetKeypointId, offsets);

            targetKeypoint = new float[]{
                targetKeypointY * mOutputStride + offsetPoint[0],
                targetKeypointX * mOutputStride + offsetPoint[1]
            };
        }

        int[] targetKeypointIndices = getStridedIndexNearPoint(targetKeypoint[0], targetKeypoint[1], height, width);

        float score = sigmoid(scores[targetKeypointIndices[0]][targetKeypointIndices[1]][targetKeypointId]);

        return makeKeypoint(score, targetKeypointId, targetKeypoint[0], targetKeypoint[1] );
    }

    private int clamp(int v, int min, int max) {
        if (v < min) return min;
        if (v > max) return max;
        return v;
    }

    private int[] getStridedIndexNearPoint(float _y, float _x, int height, int width) {
        int y = clamp(Math.round(_y / mOutputStride), 0, height - 1);
        int x = clamp(Math.round(_x / mOutputStride), 0, width - 1);
        return new int[]{y, x};
    }

    private float[] getDisplacement(int edgeId, int[] keypoint, float[][][] displacements) {
        int numEdges = displacements[0][0].length / 2;
        int y = keypoint[0];
        int x = keypoint[1];
        return new float[]{displacements[y][x][edgeId], displacements[y][x][edgeId + numEdges]};
    }

    private float getInstanceScore(List<Map<String, Object>> keypoints, List<Map<String, Object>> existingPoses, int squaredNmsRadius) {
        float scores = 0;
        for (Map<String, Object> keypoint : keypoints) {
            if (withinNmsRadiusOfCorrespondingPoint(existingPoses, squaredNmsRadius, getKeypointPosition(keypoint, "y"), getKeypointPosition(keypoint, "x"), (int) keypoint.get("partId")))
                continue;
            scores += (float) keypoint.get("score");
        }

        return scores / mNumKeypoints;
    }

    private float sigmoid(final float x) {
        return (float) (1. / (1. + Math.exp(-x)));
    }
}
