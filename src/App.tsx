import { useRef, useEffect, useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toggle } from '@/components/ui/toggle'
import { Button } from './components/ui/button'
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'
import {
  TfjsPoseDetector,
  EDGES_17,
  EDGES_33,
  type PoseDetector,
  type PoseResult,
  type Keypoint,
} from '@/lib/pose-detection/poseDetection'

function drawPoses(ctx: CanvasRenderingContext2D, poses: Keypoint[]) {
  ctx.lineWidth = 3
  ctx.strokeStyle = '#00FF7F'
  ctx.fillStyle = '#FF3B30'
  const edges = poses.length >= 33 ? EDGES_33 : EDGES_17
  // lines
  for (const [a, b] of edges) {
    const A = poses[a]
    const B = poses[b]
    if (!A || !B) continue
    if ((A.visibility ?? 0) < 0.2 || (B.visibility ?? 0) < 0.2) continue
    ctx.beginPath()
    ctx.moveTo(A.x, A.y)
    ctx.lineTo(B.x, B.y)
    ctx.stroke()
  }
  // points
  for (const p of poses) {
    if ((p.visibility ?? 0) < 0.2) continue
    ctx.beginPath()
    ctx.arc(p.x, p.y, 3, 0, Math.PI * 2)
    ctx.fill()
  }
}

function App() {
  const imgEl = useRef<HTMLImageElement | null>(null)
  const videoEl = useRef<HTMLVideoElement | null>(null)
  const canvasEl = useRef<HTMLCanvasElement | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)
  const detectorRef = useRef<PoseDetector>(new TfjsPoseDetector())
  const [inputMode, setInputMode] = useState<'IMAGE' | 'VIDEO'>('VIDEO')
  const [webcamRunning, setWebcamRunning] = useState(false)
  const webcamRunningRef = useRef(webcamRunning)
  webcamRunningRef.current = webcamRunning
  const videoFileActiveRef = useRef(false)
  const animationFrameId = useRef<number | null>(null)
  const [modelType, setModelType] = useState<'movenet-lightning' | 'blazepose-lite' | 'yolo11'>('blazepose-lite')
  const [loadingModel, setLoadingModel] = useState(false)
  const [poseResult, setPoseResult] = useState<PoseResult | null>(null)
  const poseResultHistory = useRef<PoseResult[]>([])
  const [fps, setFps] = useState('0')
  const frameCount = useRef(0)
  const lastFpsUpdateTime = useRef(performance.now())

  useEffect(() => {
    const init = async () => {
      setLoadingModel(true)

      // Clear previous session data
      if (webcamRunningRef.current) {
        await stopWebcam()
      }
      if (videoFileActiveRef.current) {
        videoFileActiveRef.current = false
        if (videoEl.current) {
          videoEl.current.pause()
          videoEl.current.removeAttribute('src')
          videoEl.current.load()
        }
      }
      if (animationFrameId.current) {
        cancelAnimationFrame(animationFrameId.current)
        animationFrameId.current = null
      }

      // Clear image
      if (imgEl.current) {
        imgEl.current.src = ''
        imgEl.current.onload = null
        imgEl.current.onerror = null
      }

      // Clear canvas
      if (canvasEl.current) {
        const ctx = canvasEl.current.getContext('2d')
        if (ctx) {
          ctx.clearRect(0, 0, canvasEl.current.width, canvasEl.current.height)
        }
      }

      // Clear file input
      if (fileInputRef.current) {
        fileInputRef.current.value = ''
      }

      // Clear pose result and history
      setPoseResult(null)
      poseResultHistory.current = []

      // Initialize detector
      await detectorRef.current.initialize({
        model: modelType,
        backend: 'webgl',
      })

      setLoadingModel(false)
    }
    init()
  }, [modelType])

  const onFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (loadingModel) {
      alert('Model is loading, please wait.')
      return
    }

    const file = e.target.files?.[0]
    if (!file || !detectorRef.current || !imgEl.current || !canvasEl.current || !videoEl.current) return

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

      const video = videoEl.current
      video.srcObject = null
      video.src = url

      video.onloadeddata = () => {
        animationFrameId.current = requestAnimationFrame(predictVideo)
      }
      video.onerror = () => {
        const e = video.error
        alert('Error loading video: ' + e?.message)
        URL.revokeObjectURL(url)
        videoFileActiveRef.current = false
      }

      video.onended = () => {
        setFps('0')
        videoFileActiveRef.current = false
      }
      return
    }

    // ---- IMAGE FILE PATH ----
    setInputMode('IMAGE')
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
    const result = await detectorRef.current.detect(img, performance.now())
    setPoseResult(result)
    poseResultHistory.current.push(result)

    ctx.save()
    drawPoses(ctx, result.keypoints)
    ctx.restore()

    URL.revokeObjectURL(url)
  }

  const predictVideo = useCallback(async () => {
    if (
      !detectorRef.current ||
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

    const result = await detectorRef.current.detect(video, performance.now())
    setPoseResult(result)
    poseResultHistory.current.push(result)

    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    drawPoses(ctx, result.keypoints)
    ctx.restore()

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
    if (loadingModel) {
      alert('Model is loading, please wait.')
      return
    }

    if (webcamRunningRef.current) {
      await stopWebcam()
    } else {
      setInputMode('VIDEO')
      await startWebcam()
    }
  }

  const exportPoseData = () => {
    if (poseResultHistory.current.length === 0) {
      alert('No data to export. Please run a webcam session first.')
      return
    }

    const jsonString = JSON.stringify({ modelType, poseResult: poseResultHistory.current }, null, 2)
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
        {/* --- Pose Detection Section --- */}
        <Label htmlFor="pose-landmarker-card" className="text-2xl font-bold">
          Pose Detection Model {loadingModel && '(Loading...)'}
        </Label>
        <RadioGroup
          className="grid-flow-col"
          defaultValue="blazepose-lite"
          orientation="horizontal"
          onValueChange={(v) => setModelType(v as any)}
          disabled={loadingModel}
        >
          <div className="flex items-center gap-3">
            <RadioGroupItem value="blazepose-lite" id="r1" />
            <Label htmlFor="r1">Mediapipe</Label>
          </div>
          <div className="flex items-center gap-3">
            <RadioGroupItem value="movenet-lightning" id="r2" />
            <Label htmlFor="r2">MoveNet</Label>
          </div>
          <div className="flex items-center gap-3">
            <RadioGroupItem value="yolo11" id="r3" />
            <Label htmlFor="r3">YOLO11</Label>
          </div>
        </RadioGroup>
        <Card id="pose-landmarker-card" className="h-full w-full px-2">
          {/** Input Source Selection **/}
          <Tabs defaultValue="file">
            <CardHeader>
              <CardTitle>Select Input Source</CardTitle>

              <CardDescription>
                <TabsList>
                  <TabsTrigger value="file" disabled={loadingModel}>
                    Image File
                  </TabsTrigger>
                  <TabsTrigger value="cam" disabled={loadingModel}>
                    Web Camera
                  </TabsTrigger>
                </TabsList>
              </CardDescription>
              <CardAction>
                <TabsContent value="file">
                  <div className="grid w-full max-w-sm gap-2">
                    <Label htmlFor="file-input">Input ({fps} FPS)</Label>
                    <Input
                      ref={fileInputRef}
                      id="file-input"
                      type="file"
                      accept="image/*,video/*"
                      onChange={onFileChange}
                      disabled={loadingModel}
                    />
                  </div>
                </TabsContent>
                <TabsContent value="cam">
                  <div className="grid w-full max-w-sm gap-2">
                    <Label>Webcam ({fps} FPS)</Label>
                    <Toggle onClick={handleWebcamClick} variant="outline" disabled={loadingModel}>
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

      {/* --- Pose Recognition Section --- */}
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        <Label htmlFor="pose-recognition-card" className="text-2xl font-bold">
          Pose Recognition Model
        </Label>
        <Card id="pose-recognition-card" className="h-full w-full px-2">
          <CardHeader>
            <CardTitle>Input</CardTitle>
            <CardAction>
              <Button onClick={exportPoseData}>Export</Button>
            </CardAction>
          </CardHeader>
          <CardContent className="bg-secondary h-full flex-1 overflow-auto p-2">
            <pre className="h-full w-full text-xs">{JSON.stringify({ modelType, poseResult }, null, 2)}</pre>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
