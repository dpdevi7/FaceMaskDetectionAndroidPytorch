package com.example.facemaskdetectionandroid;

import android.graphics.Rect;
import android.icu.text.Edits;
import android.util.Log;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;

public class NMSmine {

    private static final String TAG = "NMSmine";


    static ArrayList<Integer> demoArrayList(){
        ArrayList<Integer> results = new ArrayList<>();

        results.add(1);
        results.add(2);
        results.add(3);
        results.add(4);
        results.add(5);
        results.add(5);
        results.add(5);
        results.add(5);
        results.add(5);
        results.add(5);

        Iterator<Integer> iterator = results.iterator();

        while (iterator.hasNext()){
            int value = iterator.next();



            if (value%2 == 0){
                iterator.remove();
            }
        }

        Log.d(TAG, "demoArrayList: "+ Arrays.asList(results.toArray()));
        return results;
    }


    static ArrayList<Result> performNMS(ArrayList<Result> results, float threshold){
        // what is nms:
        // 1. box1 has iou with box2 greater than defined threshold.
        // 2. box1 has lower score than box2.
        // 3. remove box1

        // logic
        // 1. sort list in decreasing order of score.
        // 2. for 1st box --> compare with other boxes, if other n-boxes has iou > threshold and score is less, remove those boxes,
        //    update list
        // 3. for 2nd box in updated list --> compare with other boxes, if other n-boxes has iou > threshold and score is less,
        //    update list
        // 4. for 3rd box in updated list --> do same
        // 5. use arraylist iterator

        Collections.sort(results,
                new Comparator<Result>() {
                    @Override
                    public int compare(Result o1, Result o2) {
                        return o2.score.compareTo(o1.score);
                    }
                });

        boolean[] boxToConsiderFlagList = new boolean[results.size()];
        Arrays.fill(boxToConsiderFlagList, true);

        ArrayList<Result> nmsResultsBoxes = new ArrayList<>();

        Log.d(TAG, "performNMS: "+Arrays.asList(results.toArray()));

        for (int i=0 ;i< results.size() ;i++){

            // 1. if i not a last index
            if (i < results.size()-1){
                for (int j=i+1; j<results.size(); j++){

                    if (IOU(results.get(i).rect, results.get(j).rect) >= threshold){
                        boxToConsiderFlagList[j] = false;
                    }

                }
            }

            // 2. if active index, add to nmsResult
            if (boxToConsiderFlagList[i]){
                nmsResultsBoxes.add(results.get(i));
            }




        }

        Log.d(TAG, "performNMS: "+Arrays.asList(nmsResultsBoxes.toArray()));
        Log.d(TAG, "performNMS: "+nmsResultsBoxes.size());
        Log.d(TAG, "performNMS: "+results.size());
        return nmsResultsBoxes;
    }


    static float IOU(Rect a, Rect b) {
        float areaA = (a.right - a.left) * (a.bottom - a.top);
        if (areaA <= 0.0) return 0.0f;

        float areaB = (b.right - b.left) * (b.bottom - b.top);
        if (areaB <= 0.0) return 0.0f;

        float intersectionMinX = Math.max(a.left, b.left);
        float intersectionMinY = Math.max(a.top, b.top);
        float intersectionMaxX = Math.min(a.right, b.right);
        float intersectionMaxY = Math.min(a.bottom, b.bottom);
        float intersectionArea = Math.max(intersectionMaxY - intersectionMinY, 0) *
                Math.max(intersectionMaxX - intersectionMinX, 0);
        return intersectionArea / (areaA + areaB - intersectionArea);
    }
}
