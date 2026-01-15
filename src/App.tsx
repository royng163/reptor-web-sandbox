import { useRef, useEffect, useState, useCallback } from 'react'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardAction } from '@/components/ui/card'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Toggle } from '@/components/ui/toggle'
import { Button } from './components/ui/button'
import { ButtonGroup } from '@/components/ui/button-group'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuTrigger,
  DropdownMenuRadioItem,
} from '@/components/ui/dropdown-menu'
import { TransformWrapper, TransformComponent } from 'react-zoom-pan-pinch'
import { TfjsPoseDetector, EDGES_17, EDGES_33, type PoseDetector } from '@/lib/pose-detection/poseDetection'
import {
  RepDetector,
  FeatureAggregator,
  RuleEngine,
  calculateAngle,
  type Keypoint,
  type PoseResult,
  type Feedback,
  midpoint,
} from '@royng163/reptor-core'
import SQUAT_RULES from '@/assets/rules/squat_rules.json'
import { ChevronDownIcon } from 'lucide-react'

// Debug state type
interface DebugInfo {
  // Instant features
  kneeAngleLeft: number
  kneeAngleRight: number
  avgKneeAngle: number
  trunkAngle: number
  stanceWidth: number
  // Phase detection
  currentPhase: string
  repCount: number
  isRepFinished: boolean
  // Keypoint visibility
  keypointsDetected: number
  totalKeypoints: number
  // Last evaluation results
  lastEvaluation: Feedback[] | null
  avgHipY: number
  velocity: number
}

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
  const [modelType, setModelType] = useState<'movenet' | 'blazepose' | 'yolo11'>('blazepose')
  const [modelVariant, setModelVariant] = useState<'lite' | 'full' | 'heavy' | 'lightning' | 'thunder' | null>('lite')
  const [loadingModel, setLoadingModel] = useState(false)
  const [poseResult, setPoseResult] = useState<PoseResult | null>(null)
  const poseResultHistory = useRef<PoseResult[]>([])
  const [fps, setFps] = useState('0')
  const frameCount = useRef(0)
  const lastFpsUpdateTime = useRef(performance.now())

  const [feedback, setFeedback] = useState<string>('Ready to Evaluate')
  const repDetector = useRef(new RepDetector()).current
  const aggregator = useRef(new FeatureAggregator()).current
  const ruleEngine = useRef(new RuleEngine(SQUAT_RULES[0] as any)).current

  // Smoothing buffers for noisy signals
  const trunkAngleBuffer = useRef<number[]>([])

  // Debug state
  const [debugInfo, setDebugInfo] = useState<DebugInfo>({
    kneeAngleLeft: 0,
    kneeAngleRight: 0,
    avgKneeAngle: 0,
    trunkAngle: 0,
    stanceWidth: 0,
    currentPhase: 'IDLE',
    repCount: 0,
    isRepFinished: false,
    keypointsDetected: 0,
    totalKeypoints: 0,
    lastEvaluation: null,
    avgHipY: 0,
    velocity: 0,
  })

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
        variant: modelVariant ?? undefined,
      })

      setLoadingModel(false)
    }
    init()
  }, [modelType, modelVariant])

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

    // Validate required keypoints exist and have sufficient confidence
    const requiredIndices = [5, 11, 12, 13, 14, 15, 16] // shoulder_left, hips, knees, ankles
    const MIN_CONFIDENCE = 0.5 // Adjust threshold as needed

    const allKeypointsValid = requiredIndices.every((idx) => {
      const kp = result.keypoints[idx]
      return kp && kp.visibility !== undefined && kp.visibility >= MIN_CONFIDENCE
    })

    // Analyze pose for feedback
    if (allKeypointsValid) {
      // 1. Extract Instant Features
      const shoulder_left = result.keypoints[5]
      const hip_left = result.keypoints[11]
      const hip_right = result.keypoints[12]
      const knee_left = result.keypoints[13]
      const knee_right = result.keypoints[14]
      const ankle_left = result.keypoints[15]
      const ankle_right = result.keypoints[16]

      // Calculate current angles
      const kneeAngleLeft = calculateAngle(hip_left, knee_left, ankle_left)
      const kneeAngleRight = calculateAngle(hip_right, knee_right, ankle_right)
      const avgKnee = (kneeAngleLeft + kneeAngleRight) / 2
      const trunkAngle = calculateAngle(shoulder_left, hip_left, knee_left)
      const avgHip = midpoint(hip_left, hip_right)
      const stanceWidth = Math.abs(ankle_right.x - ankle_left.x)

      // 2. Detect Phase
      const { state, isRepFinished, velocity } = repDetector.detect(avgHip.y)

      // 3. Update aggregator phase and record features
      aggregator.setPhase(state)
      aggregator.processFrame(kneeAngleLeft, kneeAngleRight, trunkAngle, {
        hip_left,
        hip_right,
        ankle_left,
        ankle_right,
      })

      // 4. Evaluate Frame-level rules for instant feedback
      const frameData = {
        knee_flexion: avgKnee,
        knee_flexion_left: kneeAngleLeft,
        knee_flexion_right: kneeAngleRight,
        trunk_angle: trunkAngle,
        stance_width: stanceWidth,
      }
      const frameFeedbacks = ruleEngine.evaluateFrame(frameData, state)

      // Show debounced frame-level errors during exercise
      const frameErrors = frameFeedbacks.filter((f) => !f.passed)
      if (frameErrors.length > 0 && state !== 'IDLE') {
        setFeedback(formatErrorMessage(frameErrors[0].errorType, frameErrors[0].direction))
      } else if (state !== 'IDLE' && frameErrors.length === 0) {
        // Clear feedback when no active errors during exercise
        setFeedback('Looking Good')
      }

      // 5. Evaluate Rep-level rules (Only when rep finishes)
      if (isRepFinished) {
        const repAggregates = aggregator.getRepAggregates()
        const phaseAggregates = aggregator.getPhaseAggregates()

        // Evaluate all rules (phase + rep + accumulated frame data)
        const results = ruleEngine.evaluateWithPhases(repAggregates, phaseAggregates)

        // Update debug info with evaluation results
        setDebugInfo((prev) => ({
          ...prev,
          lastEvaluation: results,
        }))

        // Process results to find errors
        const errors = results.filter((r: Feedback) => !r.passed)

        if (errors.length === 0) {
          setFeedback('Good Rep')
        } else {
          // Show the most important error
          const msg = formatErrorMessage(errors[0].errorType, errors[0].direction)
          setFeedback(msg)
        }

        // Reset for next rep
        aggregator.reset()
        ruleEngine.reset()
        trunkAngleBuffer.current = []
      }

      // Update debug info
      setDebugInfo((prev) => ({
        ...prev,
        kneeAngleLeft,
        kneeAngleRight,
        avgKneeAngle: avgKnee,
        trunkAngle,
        stanceWidth,
        currentPhase: state,
        isRepFinished,
        totalKeypoints: result.keypoints.length,
        avgHipY: avgHip.y,
        velocity: velocity,
      }))
    }

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

  function formatErrorMessage(errorType: string, direction?: 'low' | 'high') {
    switch (errorType) {
      case 'insufficient_range_lower_body':
        return 'Go Deeper! Your squat depth is too shallow.'
      case 'joint_alignment_knees':
        if (direction === 'low') {
          return 'Widen Your Stance! Your feet are too close together.'
        } else if (direction === 'high') {
          return 'Narrow Your Stance! Your feet are too far apart.'
        }
        return 'Adjust Your Stance Width!'
      case 'balance_stability':
        return 'Stay Stable! Keep your torso steady throughout the movement.'
      case 'trunk_control':
        return 'Too Much Forward Lean! Keep your chest up and core tight.'
      default:
        return 'Check Your Form'
    }
  }

  return (
    <div className="flex flex-col gap-4 p-4 md:h-screen md:flex-row">
      <div className="flex basis-1/2 flex-col items-center justify-center gap-2">
        {/* --- Pose Detection Section --- */}
        <Label htmlFor="pose-landmarker-card" className="text-2xl font-bold">
          Pose Detection Model {loadingModel && '(Loading...)'}
        </Label>
        <ButtonGroup>
          <ButtonGroup>
            <Button
              variant={modelType === 'blazepose' ? 'default' : 'outline'}
              onClick={() => setModelType('blazepose')}
              disabled={loadingModel}
            >
              BlazePose
            </Button>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant={modelType === 'blazepose' ? 'default' : 'outline'} disabled={loadingModel}>
                  <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuRadioGroup
                  value={modelVariant as 'lite' | 'full' | 'heavy'}
                  onValueChange={(v) => {
                    setModelVariant(v as any)
                    setModelType('blazepose')
                  }}
                >
                  <DropdownMenuRadioItem value="lite">Lite</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="full">Full</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="heavy">Heavy</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </ButtonGroup>

          <ButtonGroup>
            <Button
              variant={modelType === 'movenet' ? 'default' : 'outline'}
              onClick={() => setModelType('movenet')}
              disabled={loadingModel}
            >
              MoveNet
            </Button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant={modelType === 'movenet' ? 'default' : 'outline'} disabled={loadingModel}>
                  <ChevronDownIcon />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent>
                <DropdownMenuRadioGroup
                  value={modelVariant as 'lightning' | 'thunder'}
                  onValueChange={(v) => {
                    setModelVariant(v as any)
                    setModelType('movenet')
                  }}
                >
                  <DropdownMenuRadioItem value="lightning">Lightning</DropdownMenuRadioItem>
                  <DropdownMenuRadioItem value="thunder">Thunder</DropdownMenuRadioItem>
                </DropdownMenuRadioGroup>
              </DropdownMenuContent>
            </DropdownMenu>
          </ButtonGroup>

          <ButtonGroup>
            <Button
              variant={modelType === 'yolo11' ? 'default' : 'outline'}
              onClick={() => setModelType('yolo11')}
              disabled={loadingModel}
            >
              YOLO11
            </Button>
          </ButtonGroup>
        </ButtonGroup>

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
        <Label htmlFor="rule-engine-card" className="text-2xl font-bold">
          Rule Engine
        </Label>
        <Card id="rule-engine-card" className="h-full w-full px-2">
          <CardHeader>
            <CardTitle className="text-lg">{feedback}</CardTitle>
            <CardAction>
              <Button onClick={exportPoseData}>Export</Button>
            </CardAction>
          </CardHeader>
          <CardContent className="h-full flex-1 overflow-auto p-2">
            {/* Phase Detection */}
            <div className="mb-2 text-2xl">
              <span>Current Phase:</span>
              <span className={`ml-2`}>{debugInfo.currentPhase}</span>
            </div>

            {/* Features */}
            <div className="grid grid-cols-2 gap-2 text-2xl">
              <div>
                <span>Left Knee:</span>
                <span className="ml-2">{debugInfo.kneeAngleLeft.toFixed(1)}°</span>
              </div>
              <div>
                <span>Right Knee:</span>
                <span className="ml-2">{debugInfo.kneeAngleRight.toFixed(1)}°</span>
              </div>
              <div>
                <span>Avg Knee:</span>
                <span className="ml-2">{debugInfo.avgKneeAngle.toFixed(1)}°</span>
              </div>
              <div>
                <span>Trunk:</span>
                <span className="ml-2">{debugInfo.trunkAngle.toFixed(1)}°</span>
              </div>
              <div>
                <span>Stance Width:</span>
                <span className="ml-2">{debugInfo.stanceWidth.toFixed(1)}</span>
              </div>
              <div>
                <span>Avg Hip Y:</span>
                <span className="ml-2">{debugInfo.avgHipY.toFixed(1)}</span>
              </div>

              <div>
                <span>Hip Y Velocity:</span>
                <span className="ml-2">{debugInfo.velocity.toFixed(2)}</span>
              </div>
            </div>

            {/* Last Evaluation Results */}
            {debugInfo.lastEvaluation && (
              <div className="rounded-lg">
                <div className="space-y-2 text-lg">
                  {debugInfo.lastEvaluation.map((result, idx) => (
                    <div
                      key={idx}
                      className={`flex items-center justify-between rounded p-2 ${
                        result.passed ? 'bg-green-100 dark:bg-green-900' : 'bg-red-100 dark:bg-red-900'
                      }`}
                    >
                      <span className="font-medium">{result.ruleId}</span>
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-2xl">
                          {result.value?.toFixed(2)} vs {result.threshold?.toFixed(2)}
                        </span>
                        <span>{result.passed ? '✅' : '❌'}</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

export default App
