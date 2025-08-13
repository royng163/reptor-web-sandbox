<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, nextTick } from 'vue'
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision'

const imgEl = ref<HTMLImageElement | null>(null)
const canvasEl = ref<HTMLCanvasElement | null>(null)
let landmarker: PoseLandmarker | null = null

const init = async () => {
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm',
  )
  landmarker = await PoseLandmarker.createFromOptions(vision, {
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

onMounted(() => {
  void init()
})
onBeforeUnmount(() => {
  landmarker?.close()
  landmarker = null
})

const onFileChange = async (e: Event) => {
  const file = (e.target as HTMLInputElement).files?.[0]
  if (!file || !landmarker || !imgEl.value || !canvasEl.value) return

  const url = URL.createObjectURL(file)
  // Load selected image into <img>
  await new Promise<void>((resolve, reject) => {
    imgEl.value!.onload = () => resolve()
    imgEl.value!.onerror = () => reject(new Error('Image load failed'))
    imgEl.value!.src = url
  })
  await nextTick()

  const img = imgEl.value!
  const canvas = canvasEl.value!
  const ctx = canvas.getContext('2d')!
  // Match canvas pixels to the image's natural size for correct overlay
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  ctx.clearRect(0, 0, canvas.width, canvas.height)

  // Run pose detection
  const result = landmarker.detect(img)

  // Draw pose overlay
  const drawing = new DrawingUtils(ctx)
  for (const landmarks of result.landmarks) {
    drawing.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, { color: '#00FF7F', lineWidth: 3 })
    drawing.drawLandmarks(landmarks, { color: '#FF3B30', radius: 2 })
  }

  URL.revokeObjectURL(url)
}
</script>

<template>
  <div class="flex flex-col items-center content-center h-screen">
    <input type="file" accept="image/*" @change="onFileChange" />

    <div>
      <img ref="imgEl" alt="Uploaded image preview" />
      <canvas ref="canvasEl" class="absolute top-0 left-0" />
    </div>
  </div>
</template>
