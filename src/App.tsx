import { useRef, useEffect, useState } from 'react'
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'

function App() {
  const imgEl = useRef<HTMLImageElement | null>(null)
  const canvasEl = useRef<HTMLCanvasElement | null>(null)
  const landmarker = useRef<PoseLandmarker | null>(null)
  const [inputMode, setInputMode] = useState<'IMAGE' | 'VIDEO'>('IMAGE')

  useEffect(() => {
    // https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker/web_js#create_the_task
    const createTask = async () => {
      const vision = await FilesetResolver.forVisionTasks(
        'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
      )
      landmarker.current = await PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: 'app/shared/models/pose_landmarker_lite.task',
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
    <div className="flex flex-col gap-4 p-4 md:h-screen md:flex-row">
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        <h1 className="text-2xl font-bold">Pose Landmark Detection Model</h1>
        <Card className="h-full w-full px-2">
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
                <TabsContent value="video">To Be Implemented.</TabsContent>
              </CardAction>
            </CardHeader>
          </Tabs>
          <CardContent className="bg-secondary flex-1 overflow-hidden">
            <TransformWrapper>
              <TransformComponent wrapperClass="w-full h-full" contentClass="relative h-full w-full">
                <img ref={imgEl} className="h-full w-full object-contain" />
                <canvas ref={canvasEl} className="absolute inset-0 h-full w-full" />
              </TransformComponent>
            </TransformWrapper>
          </CardContent>
        </Card>
      </div>
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        <h1 className="text-2xl font-bold">Workout Pose Correction Model</h1>
        <Card className="h-full w-full px-2">
          <CardHeader>
            <CardTitle>To Be Implemented.</CardTitle>
          </CardHeader>
          <CardContent></CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
