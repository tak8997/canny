package com.pengke.paper.scanner.processor

import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import java.util.*


const val TAG: String = "PaperProcessor"

fun processPicture(previewFrame: Mat): Corners? {
    val contours = findContours(previewFrame)
    return getCorners(contours, previewFrame.size())
}

fun cropPicture(picture: Mat, pts: List<Point>): Mat {

    pts.forEach { Log.i(TAG, "point: " + it.toString()) }
    val tl = pts[0]
    val tr = pts[1]
    val br = pts[2]
    val bl = pts[3]

    val widthA = Math.sqrt(Math.pow(br.x - bl.x, 2.0) + Math.pow(br.y - bl.y, 2.0))
    val widthB = Math.sqrt(Math.pow(tr.x - tl.x, 2.0) + Math.pow(tr.y - tl.y, 2.0))

    val dw = Math.max(widthA, widthB)
    val maxWidth = java.lang.Double.valueOf(dw)!!.toInt()


    val heightA = Math.sqrt(Math.pow(tr.x - br.x, 2.0) + Math.pow(tr.y - br.y, 2.0))
    val heightB = Math.sqrt(Math.pow(tl.x - bl.x, 2.0) + Math.pow(tl.y - bl.y, 2.0))

    val dh = Math.max(heightA, heightB)
    val maxHeight = java.lang.Double.valueOf(dh)!!.toInt()

    val croppedPic = Mat(maxHeight, maxWidth, CvType.CV_8UC4)

    val src_mat = Mat(4, 1, CvType.CV_32FC2)
    val dst_mat = Mat(4, 1, CvType.CV_32FC2)

    src_mat.put(0, 0, tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y)
    dst_mat.put(0, 0, 0.0, 0.0, dw, 0.0, dw, dh, 0.0, dh)

    val m = Imgproc.getPerspectiveTransform(src_mat, dst_mat)

    Imgproc.warpPerspective(picture, croppedPic, m, croppedPic.size())
    m.release()
    src_mat.release()
    dst_mat.release()
    Log.i(TAG, "crop finish")
    return croppedPic
}

fun enhancePicture(src: Bitmap?): Bitmap {
    val src_mat = Mat()
    Utils.bitmapToMat(src, src_mat)
    Imgproc.cvtColor(src_mat, src_mat, Imgproc.COLOR_RGBA2GRAY)
    Imgproc.adaptiveThreshold(
        src_mat,
        src_mat,
        255.0,
        Imgproc.ADAPTIVE_THRESH_MEAN_C,
        Imgproc.THRESH_BINARY,
        15,
        15.0
    )
    val result = Bitmap.createBitmap(src?.width ?: 1080, src?.height ?: 1920, Bitmap.Config.RGB_565)
    Utils.matToBitmap(src_mat, result, true)
    src_mat.release()
    return result
}

private fun findContours(src: Mat): ArrayList<MatOfPoint> {
    var approxCurve = MatOfPoint2f()
    val grayImage: Mat
    val cannedImage: Mat
    val kernel: Mat = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(9.0, 9.0))
    val dilate: Mat
    val size = Size(src.size().width, src.size().height)
    grayImage = Mat(size, CvType.CV_8UC4)
    cannedImage = Mat(size, CvType.CV_8UC1)
    dilate = Mat(size, CvType.CV_8UC1)

    Imgproc.cvtColor(src, grayImage, Imgproc.COLOR_RGB2GRAY)
    Imgproc.GaussianBlur(grayImage, grayImage, Size(5.0, 5.0), 1.0)

//    Imgproc.threshold(grayImage, grayImage, 10.0, 100.0, Imgproc.THRESH_BINARY)

    Imgproc.Canny(grayImage, cannedImage, 10.0, 120.0)
    Imgproc.dilate(cannedImage, dilate, kernel, Point(3.0, 3.0), 1)

//    Imgproc.adaptiveThreshold(grayImage, grayImage, 255.0,
//        Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C,
//        Imgproc.THRESH_BINARY,
//        (src.width() + src.height()) / 200, 1.0)

    val contours = ArrayList<MatOfPoint>()
    val hierarchy = Mat()
    Imgproc.findContours(
        dilate.clone(),
        contours,
        hierarchy,
        Imgproc.RETR_TREE,
        Imgproc.CHAIN_APPROX_SIMPLE
    )

    for ((index, cnt) in contours.withIndex()) {
        val curve = MatOfPoint2f(*cnt.toArray())
        Imgproc.approxPolyDP(curve, approxCurve, 0.1 * Imgproc.arcLength(curve, true), true)
        val numberVertices = approxCurve!!.total().toInt()

        val contourArea = Imgproc.contourArea(cnt)

        if (Math.abs(contourArea) < 100) {
            continue
        }

        //Rectangle detected
        if (numberVertices >= 4 && numberVertices <= 6) {
            val cos: MutableList<Double> = java.util.ArrayList()
            for (j in 2 until numberVertices + 1) {
                cos.add(
                    angle(
                        approxCurve!!.toArray()[j % numberVertices],
                        approxCurve!!.toArray()[j - 2],
                        approxCurve!!.toArray()[j - 1]
                    )
                )
            }
            Collections.sort(cos)
            val mincos = cos[0]
            val maxcos = cos[cos.size - 1]

            if (numberVertices == 4 && mincos >= -0.1 && maxcos <= 0.3) {

                Imgproc.drawContours(src, contours, index, Scalar(0.0, 255.0, 0.0), 3)
//                setLabel(src, "X", cnt)
            }
        }
    }

    hierarchy.release()
    grayImage.release()
    cannedImage.release()
    kernel.release()
    dilate.release()

    return contours
}

private fun setLabel(im: Mat, label: String, contour: MatOfPoint) {
    val fontface = Core.FONT_HERSHEY_SIMPLEX
    val scale = 3.0 //0.4;
    val thickness = 3 //1;
    val baseline = IntArray(1)
    val text = Imgproc.getTextSize(label, fontface, scale, thickness, baseline)
    val r = Imgproc.boundingRect(contour)
    val pt = Point(r.x + (r.width - text.width) / 2, r.y + (r.height + text.height) / 2)
    Imgproc.putText(im, label, pt, fontface, scale, Scalar(255.0, 0.0, 0.0), thickness)
}

private fun angle(pt1: Point, pt2: Point, pt0: Point): Double {
    val dx1 = pt1.x - pt0.x
    val dy1 = pt1.y - pt0.y
    val dx2 = pt2.x - pt0.x
    val dy2 = pt2.y - pt0.y
    return (dx1 * dx2 + dy1 * dy2) / Math.sqrt((dx1 * dx1 + dy1 * dy1) * (dx2 * dx2 + dy2 * dy2) + 1e-10)
}

private fun getCorners(contours: ArrayList<MatOfPoint>, size: Size): Corners? {
    val indexTo: Int
    when (contours.size) {
        in 0..5 -> indexTo = contours.size - 1
        else -> indexTo = 4
    }
    for (index in 0..contours.size) {
        if (index in 0..indexTo) {
            val c2f = MatOfPoint2f(*contours[index].toArray())
            val peri = Imgproc.arcLength(c2f, true)
            val approx = MatOfPoint2f()
            Imgproc.approxPolyDP(c2f, approx, 0.03 * peri, true)
            //val area = Imgproc.contourArea(approx)
            val points = approx.toArray().asList()
            var convex = MatOfPoint()
            approx.convertTo(convex, CvType.CV_32S);
            // select biggest 4 angles polygon
            if (points.size == 4 && Imgproc.isContourConvex(convex)){
                val foundPoints = sortPoints(points)
                return Corners(foundPoints, size)
            }
        } else {
            return null
        }
    }

    return null
}

private fun sortPoints(points: List<Point>): List<Point> {

    val p0 = points.minBy { point -> point.x + point.y } ?: Point()
    val p1 = points.minBy { point -> point.y - point.x } ?: Point()
    val p2 = points.maxBy { point -> point.x + point.y } ?: Point()
    val p3 = points.maxBy { point -> point.y - point.x } ?: Point()
    return listOf(p0, p1, p2, p3)
}