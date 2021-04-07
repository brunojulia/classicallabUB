using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;

public class Computation : MonoBehaviour
{
    // Temporary values for the number of molecules N and the sigma of the interaction
    public int N;
    public double sigma = 1;
    public double m = 1;
    public double[,] r0; 
    public double[,] v0;
    public double[,] f0;
    public int iteration = 0;

    public GameObject particleprefab;


    //The Force function uses the initial position of every particle and returns an Nx2 array with the total forces acting on each one of them
    public double[,] Force( double[,] r0)
    {
        //Here we initialize the arrays storing the position's differences (dr) and their modulus (dr2)
        //Fx and Fy are the variables used to store the net forces applied on each axys, whose values are saved in the force array
        // and reset after every for loop
        double[,,] dr = new double[N, N, 2];
        double[,] dr2 = new double[N, N];
        double Fx;
        double Fy;
        double[,] force = new double[N,2];
        //Here we compute the distance vector between every i j pair of molecules in a NxNx2 array
        for(int i=0;i < N; i++)
        {
            for(int j=0; j < N; j++)
            {
                dr[i, j, 0] = r0[i, 0] - r0[j, 0];
                dr[i, j, 1] = r0[i, 1] - r0[j, 1];
                dr2[i, j] = Math.Pow(dr[i, j, 0], 2) + Math.Pow(dr[i, j, 1], 2);

                /*if (i == j)
                {
                    Debug.Log("dr");
                    Debug.Log(dr[i, j, 0]);
                    Debug.Log("math");
                    Debug.Log(Math.Pow(dr[i, j, 0], 2));
                    Debug.Log(dr2[i, j]);
                }*/
            }
        }
        /*Debug.Log("dr2");
        Debug.Log(dr2[1, 0]);
        Debug.Log(dr2[0, 1]);
        Debug.Log("dr");
        Debug.Log(dr[0, 1,0]);
        Debug.Log(dr[1, 0,0]);*/
        //Debug.Log(dr2[1, 0]);

        //And here we compute the force with the distances determined in the previous step, derived form the WCA potential
        double lim = 1.12;
        //int count = 0;

        
        for (int i=0; i< N; i++)
        {
            Fx = 0;
            Fy = 0;
            

            
            for (int j=0; j<N; j++)
            {

                if(i != j)
                {
                    /*Debug.Log("dr");
                    Debug.Log(dr[i,j,0]);
                    Debug.Log("dr2");*/
                    //Debug.Log(dr2[i, j]);

                    if (Math.Sqrt(dr2[i, j]) < lim * sigma)
                    {
                        
                        Fx = Fx + (48* dr[i, j, 0] / dr2[i, j]) *(Math.Pow((Math.Pow(sigma, 2) / dr2[i, j]), 6) -  0.5*Math.Pow(Math.Pow(sigma, 2) / dr2[i, j], 3));
                        Fy = Fy + (48 * dr[i, j, 1] / dr2[i, j]) *(Math.Pow((Math.Pow(sigma, 2) / dr2[i, j]), 6) -  0.5*Math.Pow(Math.Pow(sigma, 2) / dr2[i, j], 3));

                        
                        /*Debug.Log("dr ij0");
                        Debug.Log(i);
                        Debug.Log(j);
                        Debug.Log("N");
                        Debug.Log(N);
                        Debug.Log(dr[i, j, 0]);
                        Debug.Log("dr2");
                        Debug.Log(dr2[i,j]);
                        Debug.Log("Pow");
                        Debug.Log(-0.5 * Math.Pow(Math.Pow(sigma, 2) / dr2[i, j], 3));
                        */
                        //Debug.Log(dr2[i, j]);
                        
                    }

                    //Debug.Log(dr2[i, j]);

                        //Debug.Log(Fx);
                }
            }
            force[i, 0] = Fx;
            force[i, 1] = Fy;

            //Debug.Log(force[i, 0]);
            /*Debug.Log("force 0");
            Debug.Log(force[i,0]);
            Debug.Log("force 1");
            Debug.Log(force[i,1]);
            */
        }
        /*Debug.Log("force");
        Debug.Log(force[0, 0]);
        Debug.Log(force[1, 0]);*/
        return force;
    }
    void Start()
    {
        //Those r0 and v0 arrays, containing respectively the positions and velocities of all the molecules, are set to 1 just for test purposes

        r0 = new double[N, 2];
        v0 = new double[N, 2];
        int sw = 1;
        double norepeat = 1;
        GameObject pref;
        int numx = 0;
        int numy = 0;
        int maxfila;
        int dist = 2;
        //int ysize = 30;
        int xsize = 50;
        maxfila = (xsize - 2) / dist;


        for (int i=0; i<N; i++)
        {
            r0[i, 0] = -25+(numx+1)*dist;
            r0[i, 1] = -13+numy / maxfila * dist;
            numx++;
            numy++;
            if (numx == maxfila)
            {
                numx = 0;
                Debug.Log("hola");
            }
            Debug.Log(r0[i, 0]);
        }

        for (int i = 0; i < N; i++)
        {
            int signx = 1;
            int signy = 1;
            if (UnityEngine.Random.value <= 0.5)
            {
                signx = -1;
            }
            if (UnityEngine.Random.value <= 0.5)
            {
                signy = -1;
            }
            v0[i, 0] = 2*signx* UnityEngine.Random.value;
            v0[i, 1] = 2*signy* UnityEngine.Random.value;
        }

        /*r0[0, 0] = 0;
        r0[0, 1] = 0;
        r0[1, 0] = 1;
        r0[1, 1] = 1;
        r0[2, 0] = 0;
        r0[2, 1] = 1;
        v0[0, 0] = 0.1;
        v0[0, 1] = 0.1;
        v0[1, 0] = -0.1;
        v0[1, 1] = -0.1;
        v0[2, 0] = -0.2;
        v0[2, 1] = 0.05;*/
        f0 = Force(r0);
        /*Debug.Log("r0");
        Debug.Log(r0[0, 0]);*/
        //Debug.Log("Force");
        //Debug.Log(f0[0, 0]);
        //Debug.Log(f0[2, 1]);
        /*particle.transform.position = new Vector2(Convert.ToSingle(r0[0,0]),Convert.ToSingle(r0[0,1]));
        particle2.transform.position = new Vector2(Convert.ToSingle(r0[1,0]), Convert.ToSingle(r0[1, 1]));
        particle3.transform.position = new Vector2(Convert.ToSingle(r0[2, 0]), Convert.ToSingle(r0[2, 1]));*/

        for(int i = 0; i < N; i++)
        {
            pref = Instantiate(particleprefab, new Vector2(Convert.ToSingle(r0[i, 0]), Convert.ToSingle(r0[i, 1])), Quaternion.identity);
            pref.name = i.ToString();
        }
    }

    
    void FixedUpdate()
    {
        //Here we fix the time step to 0.1 seconds, but again it won't be its final value, since it relies strongly on 
        // how many times the update function is called (once per every frame).
        double dt = 0.01;
        double[,] f1 = new double[N,2];
        double[,] r1 = new double[N,2];
        double[,] v1 = new double[N,2];
        GameObject partic;
        /*Debug.Log("r0");
        Debug.Log(r0[0, 0]);*/
        for(int i = 0; i < N; i++)
        {
            for(int j = 0; j < 2; j++)
            {
                r1[i, j] = r0[i, j] + dt * v0[i, j] + 0.5 * Math.Pow(dt, 2) * f0[i, j] / m;
                
            }
            partic = GameObject.Find(i.ToString());
            partic.transform.position = new Vector2(Convert.ToSingle(r1[i, 0]), Convert.ToSingle(r1[i, 1]));
        }

        f1 = Force(r1);

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < 2; j++)
            {
                v1[i, j] = v0[i, j] + 0.5 * dt / m * (f1[i, j] + f0[i, j]);
            }
        }
        
        f0 = f1;
        r0 = r1;
        v0 = v1;
        /*Debug.Log("forces");
        Debug.Log(f0[0,0])*/
        //Debug.Log(v0[0, 0]);
        //Debug.Log(v0[1, 0]);
        //particle.transform.position = new Vector2(Convert.ToSingle(r0[0, 0]), Convert.ToSingle(r0[0, 1]));
        //particle2.transform.position = new Vector2(Convert.ToSingle(r0[1, 0]), Convert.ToSingle(r0[1, 1]));
        //particle3.transform.position = new Vector2(Convert.ToSingle(r0[2, 0]), Convert.ToSingle(r0[2, 1]));

        //Debug.Log("loop");
        //Debug.Log(iteration);
        iteration++;
        /*Debug.Log("it");
        Debug.Log(iteration);*/
        //Debug.Log(r0[0, 0]);
        






    }
}
