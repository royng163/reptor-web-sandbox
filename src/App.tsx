import { useRef, useEffect, useState, useCallback } from 'react'
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toggle } from '@/components/ui/toggle'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'

function App() {
  const imgEl = useRef<HTMLImageElement | null>(null)
  const videoEl = useRef<HTMLVideoElement | null>(null)
  const canvasEl = useRef<HTMLCanvasElement | null>(null)
  const landmarker = useRef<PoseLandmarker | null>(null)
  const [inputMode, setInputMode] = useState<'IMAGE' | 'VIDEO'>('IMAGE')
  const [webcamRunning, setWebcamRunning] = useState(false)
  const webcamRunningRef = useRef(webcamRunning)
  webcamRunningRef.current = webcamRunning
  const animationFrameId = useRef<number | null>(null)
  const [poseResult, setPoseResult] = useState<any>(null)
  const [fps, setFps] = useState('0')
  const frameCount = useRef(0)
  const lastFpsUpdateTime = useRef(performance.now())

  useEffect(() => {
    // https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js#create_the_task
    const createTask = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      )
      landmarker.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'app/shared/models/pose_landmarker_lite.task',
          delegate: 'GPU',
        },
        runningMode: inputMode,
        numPoses: 1,
        minPoseDetectionConfidence: 0.5,
        minPosePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
        outputSegmentationMasks: false,
      })
    }

    void createTask()

    // Cleanup after component unmounts
    return () => {
      landmarker.current?.close()
      landmarker.current = null
    }
  }, [inputMode])

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !landmarker.current || !imgEl.current || !canvasEl.current) return

    if (webcamRunningRef.current) {
      await stopWebcam()
    }

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
    setPoseResult(result)

    // Draw pose overlay
    const drawing = new DrawingUtils(ctx)
    for (const landmarks of result.landmarks) {
      drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF7F', lineWidth: 3 })
      drawing.drawLandmarks(landmarks, { color: '#FF3B30', radius: 2 })
    }

    URL.revokeObjectURL(url)
  }

  const predictWebcam = useCallback(async () => {
    if (!landmarker.current || !videoEl.current || !canvasEl.current || !webcamRunningRef.current) {
      return
    }

    // FPS calculation
    frameCount.current++
    const now = performance.now()
    if (now >= lastFpsUpdateTime.current + 1000) {
      // Update every second
      const currentFps = frameCount.current / ((now - lastFpsUpdateTime.current) / 1000)
      setFps(currentFps.toFixed(1))
      lastFpsUpdateTime.current = now
      frameCount.current = 0
    }

    const video = videoEl.current
    const canvas = canvasEl.current
    const ctx = canvas.getContext('2d')!

    const currentTime = performance.now()
    landmarker.current.detectForVideo(video, currentTime, (result) => {
      setPoseResult({ ...result, timestamp: currentTime })
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.save()
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      const drawing = new DrawingUtils(ctx)
      for (const landmarks of result.landmarks) {
        drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF7F', lineWidth: 3 })
        drawing.drawLandmarks(landmarks, { color: '#FF3B30', radius: 2 })
      }
      ctx.restore()
    })

    // Continue the loop
    if (webcamRunningRef.current) {
      animationFrameId.current = requestAnimationFrame(predictWebcam)
    }
  }, [])

  const startWebcam = async () => {
    if (webcamRunningRef.current || !videoEl.current) return
    if (!navigator.mediaDevices?.getUserMedia) {
      console.warn('Webcam is not supported by your browser')
      return
    }

    setWebcamRunning(true)
    const constraints = {
      video: {
        // width: { ideal: 640 },
        // height: { ideal: 480 },
      },
    }
    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    videoEl.current.srcObject = stream
    videoEl.current.addEventListener('loadeddata', predictWebcam)
  }

  const stopWebcam = async () => {
    if (!webcamRunningRef.current || !videoEl.current) return

    setWebcamRunning(false)
    if (animationFrameId.current) {
      cancelAnimationFrame(animationFrameId.current)
      animationFrameId.current = null
    }

    const stream = videoEl.current.srcObject as MediaStream
    stream?.getTracks().forEach((track) => track.stop())
    videoEl.current.srcObject = null
    videoEl.current.removeEventListener('loadeddata', predictWebcam)

    // Clear canvas
    const canvas = canvasEl.current
    if (canvas) {
      const ctx = canvas.getContext('2d')
      ctx?.clearRect(0, 0, canvas.width, canvas.height)
    }
    setFps('0')
  }

  const handleWebcamClick = async () => {
    if (!landmarker.current) {
      console.log('Wait! poseLandmaker not loaded yet.')
      return
    }

    if (webcamRunningRef.current) {
      await stopWebcam()
    } else {
      await startWebcam()
    }
  }

  return (
    <div className="flex flex-col gap-4 p-4 md:h-screen md:flex-row">
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        {/* --- Pose Landmark Detection Section --- */}
        <Label htmlFor="pose-landmarker-card" className="text-2xl font-bold">
          Pose Landmark Detection Model
        </Label>
        <Card id="pose-landmarker-card" className="h-full w-full px-2">
          {/** Input Source Selection **/}
          <Tabs defaultValue="image">
            <CardHeader>
              <CardTitle>Select Input Source</CardTitle>
              <CardDescription>
                <TabsList>
                  <TabsTrigger value="image" onClick={() => setInputMode('IMAGE')}>
                    Image File
                  </TabsTrigger>
                  <TabsTrigger value="video" onClick={() => setInputMode('VIDEO')}>
                    Web Camera
                  </TabsTrigger>
                </TabsList>
              </CardDescription>
              <CardAction>
                <TabsContent value="image">
                  <div className="grid w-full max-w-sm gap-2">
                    <Label htmlFor="file-input">Input</Label>
                    <Input id="file-input" type="file" accept="image/*" onChange={onFileChange} />
                  </div>
                </TabsContent>
                <TabsContent value="video">
                  <div className="grid w-full max-w-sm gap-2">
                    <Label>Webcam ({fps} FPS)</Label>
                    <Toggle onClick={handleWebcamClick} variant="outline">
                      {webcamRunning ? 'Disable Webcam' : 'Enable Webcam'}
                    </Toggle>
                  </div>
                </TabsContent>
              </CardAction>
            </CardHeader>
          </Tabs>

          {/** Output **/}
          <CardContent className="bg-secondary flex-1 overflow-hidden">
            <TransformWrapper>
              <TransformComponent wrapperClass="w-full h-full" contentClass="relative h-full w-full">
                <img
                  ref={imgEl}
                  className={`h-full w-full object-contain ${inputMode === 'VIDEO' ? 'hidden' : 'block'}`}
                />
                <video
                  ref={videoEl}
                  autoPlay
                  className={`h-full w-full object-contain ${inputMode === 'IMAGE' ? 'hidden' : 'block'}`}
                ></video>
                <canvas ref={canvasEl} className="absolute inset-0 h-full w-full" />
              </TransformComponent>
            </TransformWrapper>
          </CardContent>
        </Card>
      </div>

      {/* --- Workout Pose Correction Section --- */}
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        <Label htmlFor="pose-correction-card" className="text-2xl font-bold">
          Workout Pose Correction Model
        </Label>
        <Card id="pose-correction-card" className="h-full w-full px-2">
          <CardHeader>
            <CardTitle>Input</CardTitle>
          </CardHeader>
          <CardContent className="bg-secondary h-full flex-1 overflow-auto p-2">
            <pre className="h-full w-full text-xs">{JSON.stringify(poseResult, null, 2)}</pre>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
