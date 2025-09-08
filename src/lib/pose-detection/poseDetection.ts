// https://github.com/tensorflow/tfjs-models/tree/master/pose-detection

import * as posedetection from '@tensorflow-models/pose-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-wasm'

export type Keypoint = {
  x: number
  y: number
  z?: number
  visibility?: number
  name?: string
}

export const EDGES_17: [number, number][] = [
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 6],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
]
export const EDGES_33: [number, number][] = [
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 12],
  [12, 24],
  [11, 23],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [29, 31],
  [30, 32],
  [11, 15],
  [12, 16],
]

export interface PoseResult {
  keypoints: Keypoint[]
  keypoints3D?: Keypoint[]
  timestamp?: number
}

export type MediaType = HTMLImageElement | HTMLVideoElement

export type TfjsModelType = 'movenet-lightning' | 'blazepose-lite' | 'posenet'

/**
 * Interface for a pose detector, abstracting away the specific implementation
 * (e.g., MediaPipe, TensorFlow.js MoveNet).
 */
export interface PoseDetector {
  /**
   * Initializes the pose detector model.
   * @param options - Selection of model and backend for TFJS models.
   */
  initialize(options?: { model?: TfjsModelType; backend?: 'webgl' | 'wasm' }): Promise<void>

  /**
   * Detects poses in a single image or video frame.
   * @param input - The image or video element to process.
   * @param timestamp - The timestamp for the frame, used in video mode.
   */
  detect(input: MediaType, timestamp: number): Promise<PoseResult>

  /**
   * Cleans up and releases model resources.
   */
  close(): Promise<void>
}

interface InitOptions {
  model?: TfjsModelType
  backend?: 'webgl' | 'wasm'
  solutionPath?: string
}

export class TfjsPoseDetector implements PoseDetector {
  private detector: posedetection.PoseDetector | null = null
  private modelType: TfjsModelType = 'blazepose-lite'
  private backend: 'webgl' | 'wasm' = 'webgl'
  private solutionPath = 'https://cdn.jsdelivr.net/npm/@mediapipe/pose'
  private initializingPromise: Promise<void> | null = null

  constructor(config?: InitOptions) {
    if (config?.model) this.modelType = config.model
    if (config?.backend) this.backend = config.backend
    if (config?.solutionPath) this.solutionPath = config.solutionPath
  }

  async initialize(options?: InitOptions) {
    if (options?.model) this.modelType = options.model
    if (options?.backend) this.backend = options.backend
    if (options?.solutionPath) this.solutionPath = options.solutionPath

    // Prevent concurrent / duplicate loads
    if (this.initializingPromise) return this.initializingPromise
    if (this.detector) {
      // If same model already loaded, skip
      return
    }

    const usingMediapipeRuntime = this.modelType === 'blazepose-lite'

    this.initializingPromise = (async () => {
      if (!usingMediapipeRuntime) {
        await tf.setBackend(this.backend)
        await tf.ready()
      }

      switch (this.modelType) {
        case 'blazepose-lite': {
          this.detector = await posedetection.createDetector(posedetection.SupportedModels.BlazePose, {
            runtime: 'mediapipe',
            modelType: 'lite',
            solutionPath: this.solutionPath,
          })
          break
        }
        case 'movenet-lightning': {
          this.detector = await posedetection.createDetector(posedetection.SupportedModels.MoveNet, {
            modelType: posedetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
          })
          break
        }
        case 'posenet': {
          this.detector = await posedetection.createDetector(posedetection.SupportedModels.PoseNet)
          break
        }
        default:
          throw new Error('Unsupported model')
      }
    })()

    try {
      await this.initializingPromise
    } finally {
      this.initializingPromise = null
    }
  }

  async detect(input: MediaType, timestamp?: number): Promise<PoseResult> {
    if (!this.detector) {
      console.warn('Detector not initialized, returning empty result.')
      return { keypoints: [], keypoints3D: [], timestamp }
    }
    const poses = await this.detector.estimatePoses(input, { flipHorizontal: false })

    if (!poses || poses.length === 0) {
      return { keypoints: [], keypoints3D: [], timestamp }
    }

    const p = poses[0]

    const estimatedKeypoints = p.keypoints.map((kp) => ({
      x: kp.x,
      y: kp.y,
      z: kp.z,
      visibility: kp.score,
      name: kp.name,
    }))

    const estimatedKeypoints3D =
      p.keypoints3D?.map((kp) => ({
        x: kp.x,
        y: kp.y,
        z: kp.z,
        visibility: kp.score,
        name: kp.name,
      })) || []

    return {
      keypoints: estimatedKeypoints,
      keypoints3D: estimatedKeypoints3D,
      timestamp,
    }
  }

  async close() {
    this.detector?.dispose()
    this.detector = null
  }
}
