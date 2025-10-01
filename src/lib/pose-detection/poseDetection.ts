// https://github.com/tensorflow/tfjs-models/tree/master/pose-detection

import * as posedetection from '@tensorflow-models/pose-detection'
import * as tf from '@tensorflow/tfjs-core'
import '@tensorflow/tfjs-backend-webgl'
import '@tensorflow/tfjs-backend-wasm'
import { loadGraphModel, type GraphModel } from '@tensorflow/tfjs-converter'
import { computeLetterbox, mapFromLetterbox } from '@royng163/reptor-core'

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

export type TfjsModelType = 'movenet-lightning' | 'blazepose-lite' | 'yolo11'

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
  private yoloDetector: GraphModel | null = null
  private modelType: TfjsModelType = 'blazepose-lite'
  private loadedModelType: TfjsModelType | null = null
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
    if (this.loadedModelType === this.modelType) {
      // If same model already loaded, skip
      return
    }

    const usingMediapipeRuntime = this.modelType === 'blazepose-lite'

    this.initializingPromise = (async () => {
      if (!usingMediapipeRuntime) {
        await tf.setBackend(this.backend)
        await tf.ready()
      }

      console.log(`Loading model: ${this.modelType} ...`)
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
        case 'yolo11': {
          this.yoloDetector = await loadGraphModel('/models/yolo11/model.json')
          this.detector = null
          break
        }
        default:
          throw new Error('Unsupported model')
      }

      this.loadedModelType = this.modelType
      console.log(`Model loaded: ${this.modelType}`)
    })()

    try {
      await this.initializingPromise
    } finally {
      this.initializingPromise = null
    }
  }

  async detect(input: MediaType, timestamp?: number): Promise<PoseResult> {
    if (!this.detector && !this.yoloDetector) {
      console.warn('Detector not initialized, returning empty result.')
      return { keypoints: [], keypoints3D: [], timestamp }
    }

    if (this.modelType === 'yolo11') {
      return this.detectYolo11(input, timestamp)
    }

    const poses = await this.detector!.estimatePoses(input, { flipHorizontal: false })

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

  private async detectYolo11(input: MediaType, timestamp?: number): Promise<PoseResult> {
    const INPUT_SIZE = 640 // YOLO input size
    const srcW = input instanceof HTMLVideoElement ? input.videoWidth : input.naturalWidth
    const srcH = input instanceof HTMLVideoElement ? input.videoHeight : input.naturalHeight

    const letter = computeLetterbox(srcW, srcH, INPUT_SIZE)

    const tensor4d = tf.tidy(() => {
      // Create tensor [H, W, 3]
      const img = tf.browser.fromPixels(input as HTMLImageElement | HTMLVideoElement)
      const resized = tf.image.resizeBilinear(img, [letter.resized.height, letter.resized.width], true)
      const padded = tf.pad(resized, [
        [letter.dy, INPUT_SIZE - letter.resized.height - letter.dy],
        [letter.dx, INPUT_SIZE - letter.resized.width - letter.dx],
        [0, 0],
      ])
      const normalized = tf.div(padded, 255)
      return tf.expandDims(normalized, 0) as tf.Tensor4D // [1, S, S, 3]
    })

    try {
      // Run model (single input)
      const out = this.yoloDetector!.execute(tensor4d) as tf.Tensor // shape [1, C, N] or [1, N, C]
      const transpose = tf.transpose(out, [0, 2, 1]) // [1, N, C] -> easier slicing

      // Boxes: convert [cx,cy,w,h] to [y1,x1,y2,x2]
      const boxes = tf.tidy(() => {
        const w = tf.slice(transpose, [0, 0, 2], [-1, -1, 1])
        const h = tf.slice(transpose, [0, 0, 3], [-1, -1, 1])
        const x1 = tf.sub(tf.slice(transpose, [0, 0, 0], [-1, -1, 1]), tf.div(w, 2))
        const y1 = tf.sub(tf.slice(transpose, [0, 0, 1], [-1, -1, 1]), tf.div(h, 2))
        return tf.squeeze(tf.concat([y1, x1, tf.add(y1, h), tf.add(x1, w)], 2))
      }) as tf.Tensor2D // [N,4]

      const scores = tf.squeeze(tf.slice(transpose, [0, 0, 4], [-1, -1, 1])) as tf.Tensor1D // [N]
      const landmarks = tf.squeeze(tf.slice(transpose, [0, 0, 5], [-1, -1, -1])) as tf.Tensor2D // [N, 3*17]
      const selected = await tf.image.nonMaxSuppressionAsync(boxes, scores, 50, 0.45, 0.3)
      const idxArr = await selected.array()
      const topIdx = idxArr[0]
      // Take top detection’s keypoints and reshape to [17,3] = [x,y,v]
      const kpTensor = tf.reshape(tf.gather(landmarks, topIdx), [17, 3]) // [17,3]
      const kpArr = (await kpTensor.array()) as number[][]

      const kpts: Keypoint[] = kpArr!.map(([x, y, v]) => {
        // If the model outputs normalized coords (0..1), we map with normalized=true
        const normalized = x <= 1 && y <= 1
        const mapped = mapFromLetterbox(x, y, srcW, srcH, letter, normalized)
        return { x: mapped.x, y: mapped.y, visibility: v }
      })
      return { keypoints: kpts, keypoints3D: [], timestamp }
    } finally {
      tensor4d.dispose()
      await tf.nextFrame() // let TFJS flush GPU
    }
  }

  async close() {
    this.detector?.dispose()
    this.detector = null
  }
}
