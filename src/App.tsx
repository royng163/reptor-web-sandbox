import { useRef, useEffect } from 'react'
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'

function App() {
  const imgEl = useRef<HTMLImageElement | null>(null)
  const canvasEl = useRef<HTMLCanvasElement | null>(null)
  const landmarker = useRef<PoseLandmarker | null>(null)

  // https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js#configuration_options
  useEffect(() => {
    const init = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
      )
      landmarker.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'app/shared/models/pose_landmarker_lite.task',
        },
        runningMode: 'IMAGE',
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      })
    }

    void init()

    return () => {
      landmarker.current?.close()
      landmarker.current = null
    }
  }, [])

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !landmarker.current || !imgEl.current || !canvasEl.current) return

    const url = URL.createObjectURL(file)
    // Load selected image into <img>
    await new Promise<void>((resolve, reject) => {
      if (!imgEl.current) {
        reject(new Error('Image element not found'))
        return
      }
      imgEl.current.onload = () => resolve()
      imgEl.current.onerror = () => reject(new Error('Image load failed'))
      imgEl.current.src = url
    })

    const img = imgEl.current!
    const canvas = canvasEl.current!
    const ctx = canvas.getContext('2d')!
    // Match canvas pixels to the image's natural size for correct overlay
    canvas.width = img.naturalWidth
    canvas.height = img.naturalHeight
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    // Run pose detection
    const result = landmarker.current.detect(img)

    // Draw pose overlay
    const drawing = new DrawingUtils(ctx)
    for (const landmarks of result.landmarks) {
      drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF7F', lineWidth: 3 })
      drawing.drawLandmarks(landmarks, { color: '#FF3B30', radius: 2 })
    }

    URL.revokeObjectURL(url)
  }

  return (
    <div className="flex flex-col items-center content-center h-screen">
      <input type="file" accept="image/*" onChange={onFileChange} />
      <div>
        <img ref={imgEl} alt="Uploaded image preview" />
        <canvas ref={canvasEl} className="absolute top-0 left-0" />
      </div>
    </div>
  )
}

export default App
