package com.example.facemaskdetectionandroid;

import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;


import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Rect;
import android.hardware.camera2.CameraManager;
import android.os.Build;
import android.os.Bundle;
import android.text.TextUtils;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.facebook.soloader.nativeloader.NativeLoader;
import com.facebook.soloader.nativeloader.SystemDelegate;
import com.google.android.material.button.MaterialButton;


import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.PyTorchAndroid;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.concurrent.TimeUnit;

public class MainActivity extends AppCompatActivity {

    private Module mModule = null;
    private ResultView resultView;
    private ImageView imageView;
    private TextView textView;
    private MaterialButton materialButton;
    private Bitmap imageBitmap;
    private String path="test.jpg";
    private static final String TAG = "MainActivity";
    private Tensor outputTensor;
    private DecimalFormat decimalFormat = new DecimalFormat("#.####");
    private float mImgScaleX, mImgScaleY, mIvScaleX, mIvScaleY, mStartX, mStartY;

    // This One is Must for including binaries of vision
    static {
        if (!NativeLoader.isInitialized()) {
            NativeLoader.init(new SystemDelegate());
        }
        NativeLoader.loadLibrary("pytorch_jni");
        NativeLoader.loadLibrary("torchvision_ops");
    }

    private static final HashMap<Integer, String> intClassHashMap = new HashMap<Integer, String>();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        // Inflate Layout
        infalteLayout();

        getIntClassHashMap();

        // get Image in Bitmap and Set to ImageView
        imageBitmap = loadImage("4.jpg");
        imageView.setImageBitmap(imageBitmap);

        try {
            mModule = getObjectDetectionModelFromAsset("model.torchscript.pt");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
            finish();
        }

        // Detect masks
        materialButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                imageBitmap = drawBBOX(detectMaskFromImage(), imageBitmap);
                imageView.setImageBitmap(imageBitmap);

            }
        });



    }

    private void getIntClassHashMap() {
        intClassHashMap.put(0,"background");
        intClassHashMap.put(1,"with mask");
        intClassHashMap.put(2,"without mask");
        intClassHashMap.put(3,"mask weared incorrect");
    }

    private Bitmap drawBBOX(ArrayList<Result> results, Bitmap bitmap) {

        for (Result result: results){

            if (result.score > 0.1f){

                Log.d(TAG, "drawBBOX: "+result.toString());
                Log.d(TAG, "drawBBOX:height "+bitmap.getHeight());
                Log.d(TAG, "drawBBOX:width "+bitmap.getWidth());

                Canvas canvas = new Canvas(bitmap);
                // painting options for the rectangles
                Paint paint = new Paint();
                paint.setAlpha(0xA0); // the transparency
                paint.setColor(Color.RED); // color is red
                paint.setStyle(Paint.Style.STROKE); // stroke or fill or ...
                paint.setStrokeWidth(5); // the stroke width
                // The rectangle will be draw / painted from point (0,0) to point (10,20).
                // draw that rectangle on the canvas
                canvas.drawRect(result.rect, paint);
                // create Paint Object for Writing text on bbox
                Paint textPaint = new Paint();
                textPaint.setColor(Color.WHITE);
                textPaint.setTextSize((float) 20.9);
                textPaint.setFakeBoldText(true);
                textPaint.setTextSize(20);

                String classLabel = intClassHashMap.get(result.classIndex);
                String message = classLabel ;// + " " + decimalFormat.format(result.score) + result.rect.toString();

                canvas.drawText(message, (result.rect.left + result.rect.right)/2.0f, (result.rect.top+result.rect.bottom)/2.0f, textPaint);
                imageView.setImageBitmap(bitmap);
//                try {
//                    TimeUnit.SECONDS.sleep(2);
//                } catch (InterruptedException e) {
//                    e.printStackTrace();
//                }

            }
        }

        return bitmap ;

    }


    private ArrayList<Result> detectMaskFromImage() {

        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(imageBitmap,
                PrePostProcessor.NO_MEAN_RGB,
                PrePostProcessor.NO_STD_RGB);

        final Tensor[] outputTensor = mModule.forward(IValue.from(inputTensor)).toTensorList();

        final float[] bboxList = outputTensor[0].getDataAsFloatArray();
        final float[] scoreList = outputTensor[1].getDataAsFloatArray();
        final long[] classList = outputTensor[2].getDataAsLongArray();

        ArrayList<Result> resultArrayList = new ArrayList<>();

        for (int i=0, j=0, k=0 ; i<bboxList.length && j<scoreList.length && k<classList.length ; i=i+3, j++, k++){

            Rect rect = new Rect();
            rect.left = (int) bboxList[i];
            rect.top = (int) bboxList[i+1];
            rect.right = (int) bboxList[i+2];
            rect.bottom = (int) bboxList[i+3];

            int cls = (int) classList[k];
            float score = scoreList[j];

            Result result = new Result(cls, score, rect);
            resultArrayList.add(result);
        }



//         ArrayList<Result> resultsNMS = PrePostProcessor.nonMaxSuppression(resultArrayList, 200, (float) 0.0);

        return NMSmine.performNMS(resultArrayList, 0.000000000000000000001f);
    }


    private Bitmap loadImage(String imageFile){
        Bitmap bitmap = null;
        try {
            BitmapFactory.Options options = new BitmapFactory.Options();
            options.inScaled = false;
            options.inMutable = true;
            bitmap  = BitmapFactory.decodeStream(getAssets().open(imageFile));
            bitmap = bitmap.copy(Bitmap.Config.RGB_565, true);


        } catch (IOException e) {
            e.printStackTrace();
            Toast.makeText(MainActivity.this, "BitMap Null", Toast.LENGTH_LONG).show();
            return bitmap;
        }

        return bitmap;
    }

    private void infalteLayout() {
        imageView = findViewById(R.id.mainActivity_iv);
        textView = findViewById(R.id.mainActivity_tv);
        materialButton = findViewById(R.id.mainActivity_mbtn);
        resultView = findViewById(R.id.mainactivity_rv);
    }

    private Module getObjectDetectionModelFromAsset(String fileName) throws FileNotFoundException {
        Module module = PyTorchAndroid.loadModuleFromAsset(getAssets(), fileName);
        return module;
    }

}