import { useRef, useEffect, useState, useCallback } from 'react'
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toggle } from '@/components/ui/toggle'
import { Button } from './components/ui/button'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'

function drawPoses(ctx: CanvasRenderingContext2D, landmarksGroups: any[][], visibilityThreshold: number = 0.0) {
  const drawing = new DrawingUtils(ctx)
  for (const landmarks of landmarksGroups) {
    // Collect visible indices
    const visible = new Set<number>()
    landmarks.forEach((lm: any, i: number) => {
      if ((lm.visibility ?? 0) >= visibilityThreshold) visible.add(i)
    })
    const visibleLandmarks = Array.from(visible.values()).map((i) => landmarks[i])
    drawing.drawConnectors(visibleLandmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF7F', lineWidth: 3 })
    drawing.drawLandmarks(visibleLandmarks, { color: '#FF3B30', radius: 2 })
  }
}

function App() {
  const imgEl = useRef<HTMLImageElement | null>(null)
  const videoEl = useRef<HTMLVideoElement | null>(null)
  const canvasEl = useRef<HTMLCanvasElement | null>(null)
  const landmarker = useRef<PoseLandmarker | null>(null)
  const [inputMode, setInputMode] = useState<'IMAGE' | 'VIDEO'>('VIDEO')
  const [webcamRunning, setWebcamRunning] = useState(false)
  const webcamRunningRef = useRef(webcamRunning)
  webcamRunningRef.current = webcamRunning
  const videoFileActiveRef = useRef(false)
  const animationFrameId = useRef<number | null>(null)
  const [poseResult, setPoseResult] = useState<any>(null)
  const poseResultHistory = useRef<any[]>([])
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
        runningMode: 'VIDEO',
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
  }, [])

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !landmarker.current || !imgEl.current || !canvasEl.current || !videoEl.current) return

    // Clear previous session data
    if (webcamRunningRef.current) {
      await stopWebcam()
    }
    if (videoFileActiveRef.current) {
      videoFileActiveRef.current = false
      const v = videoEl.current
      v.pause()
      v.removeAttribute('src')
      v.load()
    }
    poseResultHistory.current = []
    setPoseResult(null)

    const url = URL.createObjectURL(file)
    // ---- VIDEO FILE PATH ----
    if (file.type.startsWith('video/')) {
      videoFileActiveRef.current = true
      setInputMode('VIDEO')
      console.log('Set runningMode to VIDEO')
      await landmarker.current.setOptions({ runningMode: 'VIDEO' })

      const video = videoEl.current
      video.srcObject = null
      video.src = url
      video.onloadeddata = () => {
        animationFrameId.current = requestAnimationFrame(predictVideo)
      }
      video.onended = () => {
        videoFileActiveRef.current = false
      }
      return
    }

    // ---- IMAGE FILE PATH ----
    setInputMode('IMAGE')
    await landmarker.current.setOptions({ runningMode: 'IMAGE' })
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
    poseResultHistory.current.push(result)

    ctx.save()
    drawPoses(ctx, result.landmarks)
    ctx.restore()

    URL.revokeObjectURL(url)
  }

  const predictVideo = useCallback(async () => {
    if (
      !landmarker.current ||
      !videoEl.current ||
      !canvasEl.current ||
      (!webcamRunningRef.current && !videoFileActiveRef.current)
    ) {
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
      const resultWithTimestamp = { ...result, timestamp: currentTime }
      setPoseResult(resultWithTimestamp)
      poseResultHistory.current.push(resultWithTimestamp)
      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.save()
      ctx.clearRect(0, 0, canvas.width, canvas.height)
      drawPoses(ctx, result.landmarks)
      ctx.restore()
    })

    // Continue the loop
    if (webcamRunningRef.current || videoFileActiveRef.current) {
      animationFrameId.current = requestAnimationFrame(predictVideo)
    }
  }, [])

  const startWebcam = async () => {
    if (webcamRunningRef.current || !videoEl.current) return
    if (!navigator.mediaDevices?.getUserMedia) {
      console.warn('Webcam is not supported by your browser')
      return
    }

    // Clear previous session data
    poseResultHistory.current = []
    setPoseResult(null)

    setWebcamRunning(true)
    const constraints = {
      video: {
        // width: { ideal: 640 },
        // height: { ideal: 480 },
      },
    }
    const stream = await navigator.mediaDevices.getUserMedia(constraints)
    videoEl.current.srcObject = stream
    videoEl.current.addEventListener('loadeddata', predictVideo)
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
    videoEl.current.removeEventListener('loadeddata', predictVideo)

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
      setInputMode('VIDEO')
      await landmarker.current.setOptions({ runningMode: 'VIDEO' })
      await startWebcam()
    }
  }

  const exportPoseData = () => {
    if (poseResultHistory.current.length === 0) {
      alert('No data to export. Please run a webcam session first.')
      return
    }

    const jsonString = JSON.stringify(poseResultHistory.current, null, 2)
    const blob = new Blob([jsonString], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `pose_session_${Date.now()}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
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
          <Tabs defaultValue="file">
            <CardHeader>
              <CardTitle>Select Input Source</CardTitle>
              <CardDescription>
                <TabsList>
                  <TabsTrigger value="file">Image File</TabsTrigger>
                  <TabsTrigger value="cam">Web Camera</TabsTrigger>
                </TabsList>
              </CardDescription>
              <CardAction>
                <TabsContent value="file">
                  <div className="grid w-full max-w-sm gap-2">
                    <Label htmlFor="file-input">Input ({fps} FPS)</Label>
                    <Input id="file-input" type="file" accept="image/*,video/*" onChange={onFileChange} />
                  </div>
                </TabsContent>
                <TabsContent value="cam">
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
                  className={`h-full w-full object-contain ${inputMode == 'IMAGE' ? 'block' : 'hidden'}`}
                />
                <video
                  ref={videoEl}
                  autoPlay
                  className={`h-full w-full object-contain ${inputMode == 'VIDEO' ? 'block' : 'hidden'}`}
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
            <CardAction>
              <Button onClick={exportPoseData}>Export</Button>
            </CardAction>
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
